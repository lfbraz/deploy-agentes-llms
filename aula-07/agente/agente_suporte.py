"""
agente_suporte.py — o agente embrulhado num MLflow `ResponsesAgent` (Databricks, ao vivo)
=========================================================================================

Este módulo costura tudo num único agente, seguindo o caminho recomendado pela documentação
do **Databricks Agent Framework**: autorar o agente com a interface **`ResponsesAgent`** do
MLflow para ganhar, de graça, *tracing*, avaliação, deploy e monitoramento integrados ao
workspace. O agente reúne os pilares das aulas:

    ENTRADA → [guardrail de entrada] → [agente: RAG + tool use] → [guardrail de saída] → SAÍDA
                                              │
                                       (observabilidade: trace + telemetria + feedback)

**Não há modo offline.** O "cérebro" é sempre um endpoint Foundation Model (Claude) do
Databricks, via `ChatDatabricks`, orquestrado por um grafo LangGraph e servido pela classe
`AgenteSuporteResponses(ResponsesAgent)`. Executar `responder()` exige `DATABRICKS_HOST`/
`DATABRICKS_TOKEN` e as libs `databricks-langchain` e `langgraph` — na falta de qualquer um,
o agente levanta um erro didático (não há fallback simulado).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from .config import CONFIG_PADRAO, ConfigAgente
from .guardrails import guardrail_entrada, guardrail_saida
from .mlflow_databricks import configurar_mlflow, exigir_databricks


# ---------------------------------------------------------------------------------------
# Resultado e telemetria
# ---------------------------------------------------------------------------------------

@dataclass
class Resposta:
    """Saída do agente + a observabilidade daquela requisição (uma 'trace')."""

    pergunta: str
    texto: str
    trace_id: str
    ferramentas_usadas: list[str] = field(default_factory=list)
    bloqueado: bool = False
    motivo_guardrail: str = "ok"
    latencia_ms: float = 0.0
    tokens: int = 0


# Telemetria acumulada no processo (as traces "oficiais" ficam no MLflow do workspace).
TELEMETRIA: list[Resposta] = []
FEEDBACKS: list[dict] = []


def limpar_telemetria() -> None:
    TELEMETRIA.clear()
    FEEDBACKS.clear()


# ---------------------------------------------------------------------------------------
# Agente "vivo" no Databricks — MLflow ResponsesAgent + LangGraph
# ---------------------------------------------------------------------------------------
# A classe segue o padrão de referência do MLflow para agentes LangGraph (compatível com o
# Mosaic AI Agent Framework, AI Playground, Agent Evaluation e Model Serving).

import mlflow  # noqa: E402  (mlflow é dependência da Aula 07)
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    output_to_responses_items_stream,
    to_chat_completions_input,
)


class AgenteSuporteResponses(ResponsesAgent):
    """Embrulha um grafo LangGraph na interface ResponsesAgent (predict / predict_stream).

    O `ResponsesAgent` já rastreia `predict`/`predict_stream` automaticamente no MLflow.
    Cada passo do grafo (LLM, ferramentas) vira um span graças ao `mlflow.langchain.autolog()`
    chamado em `construir_agente_databricks`.
    """

    def __init__(self, grafo):
        self.grafo = grafo

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        itens = [ev.item for ev in self.predict_stream(request)
                 if ev.type == "response.output_item.done"]
        return ResponsesAgentResponse(output=itens, custom_outputs=request.custom_inputs)

    def predict_stream(self, request: ResponsesAgentRequest):
        # Converte a entrada (formato Responses) para mensagens estilo chat-completions.
        msgs = to_chat_completions_input([i.model_dump() for i in request.input])
        for _, eventos in self.grafo.stream({"messages": msgs}, stream_mode=["updates"]):
            for dados_no in eventos.values():
                yield from output_to_responses_items_stream(chunks=dados_no["messages"])


def construir_agente_databricks(config: ConfigAgente = CONFIG_PADRAO) -> AgenteSuporteResponses:
    """Monta o agente vivo: ChatDatabricks (Claude) + ferramentas + LangGraph."""
    exigir_databricks(libs=("databricks_langchain", "langgraph"))
    configurar_mlflow()
    from databricks_langchain import ChatDatabricks
    from langgraph.prebuilt import create_react_agent

    from .ferramentas import ferramentas_langchain

    mlflow.langchain.autolog()      # tracing automático de cada passo do agente
    llm = ChatDatabricks(endpoint=config.endpoint, temperature=config.temperatura,
                         max_tokens=config.max_tokens)
    grafo = create_react_agent(llm, tools=ferramentas_langchain(), prompt=config.system_prompt)
    return AgenteSuporteResponses(grafo)


def _texto_de_responses(resp: ResponsesAgentResponse) -> str:
    """Extrai o texto final dos itens de saída de um ResponsesAgentResponse."""
    partes: list[str] = []
    for item in getattr(resp, "output", None) or []:
        d = item if isinstance(item, dict) else item.model_dump()
        if d.get("type") == "message":
            for parte in d.get("content", []):
                if parte.get("type") in ("output_text", "text"):
                    partes.append(parte.get("text", ""))
        elif d.get("type") in ("output_text", "text"):
            partes.append(d.get("text", ""))
    return " ".join(t for t in partes if t).strip()


def _ferramentas_de_responses(resp: ResponsesAgentResponse) -> list[str]:
    """Lista as ferramentas que o agente chamou (itens function_call da trajetória)."""
    nomes = []
    for item in getattr(resp, "output", None) or []:
        d = item if isinstance(item, dict) else item.model_dump()
        if d.get("type") == "function_call":
            nomes.append(d.get("name", "ferramenta"))
    return nomes


# Cache do agente vivo, por endpoint (construído uma vez por processo/configuração).
_AGENTES: dict[str, AgenteSuporteResponses] = {}


def _agente(config: ConfigAgente) -> AgenteSuporteResponses:
    if config.endpoint not in _AGENTES:
        _AGENTES[config.endpoint] = construir_agente_databricks(config)
    return _AGENTES[config.endpoint]


# ---------------------------------------------------------------------------------------
# Interface pública: o pipeline completo (guardrails + agente + observabilidade)
# ---------------------------------------------------------------------------------------

def responder(pergunta: str, config: ConfigAgente = CONFIG_PADRAO) -> Resposta:
    """Executa o pipeline completo (sempre no Databricks) e devolve a `Resposta` + telemetria."""
    t0 = time.perf_counter()

    # (1) Guardrail de ENTRADA — barra jailbreak/PII ANTES de gastar uma chamada de LLM.
    ge = guardrail_entrada(pergunta)
    if ge.bloqueado:
        resp = Resposta(pergunta, "Desculpe, não posso ajudar com esse pedido.",
                        trace_id=f"tr-local-{uuid.uuid4().hex[:8]}", ferramentas_usadas=[],
                        bloqueado=True, motivo_guardrail=ge.motivo,
                        latencia_ms=(time.perf_counter() - t0) * 1000, tokens=0)
        TELEMETRIA.append(resp)
        return resp

    # (2) Agente vivo no Databricks (RAG + tool use), com tracing automático.
    saida = _agente(config).predict(
        ResponsesAgentRequest(input=[{"role": "user", "content": ge.texto}]))
    texto = _texto_de_responses(saida)
    ferramentas = _ferramentas_de_responses(saida)
    trace_id = mlflow.get_last_active_trace_id() or f"tr-{uuid.uuid4().hex[:12]}"

    # (3) Guardrail de SAÍDA — impede vazamento de PII/segredo na resposta.
    gs = guardrail_saida(texto)
    texto_final = gs.texto if (gs.bloqueado or gs.sinais) else texto
    motivo = ge.motivo if ge.sinais else gs.motivo

    resp = Resposta(
        pergunta=pergunta, texto=texto_final, trace_id=trace_id,
        ferramentas_usadas=ferramentas, bloqueado=gs.bloqueado, motivo_guardrail=motivo,
        latencia_ms=(time.perf_counter() - t0) * 1000,
        tokens=max(1, (len(ge.texto) + len(texto_final)) // 4),
    )
    TELEMETRIA.append(resp)
    return resp


def registrar_feedback(resp: Resposta, util: bool, comentario: str = "") -> None:
    """Registra feedback do usuário (👍/👎) como uma *assessment* ligada à trace no MLflow."""
    FEEDBACKS.append({"trace_id": resp.trace_id, "util": util, "comentario": comentario})
    try:
        mlflow.log_feedback(trace_id=resp.trace_id, name="user_thumbs",
                            value=bool(util), rationale=comentario or None)
    except Exception:
        pass
