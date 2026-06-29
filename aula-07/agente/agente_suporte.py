"""
agente_suporte.py — o agente embrulhado num MLflow `ResponsesAgent`
===================================================================

Costura tudo num único agente, seguindo o caminho recomendado pelo **Databricks Agent
Framework**: autorar com a interface **`ResponsesAgent`** do MLflow para ganhar *tracing*,
avaliação, deploy e monitoramento integrados. O agente reúne os pilares das aulas:

    ENTRADA → [guardrail de entrada] → [agente: RAG + tool use] → [guardrail de saída] → SAÍDA
                                              │
                                       (observabilidade: trace + telemetria + feedback)

O "cérebro" é um endpoint Foundation Model (Claude) via `ChatDatabricks`. O *tool use* é um
**loop de decisão explícito** (`bind_tools` → o modelo pede ferramentas → executamos →
devolvemos a observação até ele concluir), como na Aula 4. O pipeline público é `responder()`.
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


TELEMETRIA: list[Resposta] = []
FEEDBACKS: list[dict] = []


def limpar_telemetria() -> None:
    TELEMETRIA.clear()
    FEEDBACKS.clear()


# ---------------------------------------------------------------------------------------
# Cérebro: ChatDatabricks + loop de decisão (tool use)
# ---------------------------------------------------------------------------------------

import mlflow  # noqa: E402  (mlflow é dependência da Aula 07)

# Cache do LLM (com ferramentas) por endpoint — construído uma vez por processo/configuração.
_LLMS: dict[str, tuple] = {}


def _construir_llm(config: ConfigAgente):
    """Monta o ChatDatabricks com as ferramentas vinculadas e o registro nome->ferramenta."""
    exigir_databricks(libs=("databricks_langchain",))
    configurar_mlflow()
    from databricks_langchain import ChatDatabricks

    from .ferramentas import ferramentas_langchain

    mlflow.langchain.autolog()      # tracing automático de cada chamada ao modelo/ferramenta
    llm = ChatDatabricks(endpoint=config.endpoint, temperature=config.temperatura,
                         max_tokens=config.max_tokens)
    tools = ferramentas_langchain()
    return llm.bind_tools(tools), {t.name: t for t in tools}


def _llm(config: ConfigAgente):
    if config.endpoint not in _LLMS:
        _LLMS[config.endpoint] = _construir_llm(config)
    return _LLMS[config.endpoint]


def _loop_decisao(config: ConfigAgente, pergunta: str, max_voltas: int = 5) -> tuple[str, list[str]]:
    """Loop de tool use: o modelo decide chamar ferramentas; executamos até ele concluir."""
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    llm_tools, registro = _llm(config)
    mensagens = [SystemMessage(config.system_prompt), HumanMessage(pergunta)]
    ferramentas_usadas: list[str] = []

    resposta = llm_tools.invoke(mensagens)
    mensagens.append(resposta)
    voltas = 0
    while getattr(resposta, "tool_calls", None) and voltas < max_voltas:
        for tc in resposta.tool_calls:
            ferramentas_usadas.append(tc["name"])
            observacao = registro[tc["name"]].invoke(tc["args"])
            mensagens.append(ToolMessage(str(observacao), tool_call_id=tc["id"]))
        resposta = llm_tools.invoke(mensagens)
        mensagens.append(resposta)
        voltas += 1

    texto = resposta.content if isinstance(resposta.content, str) else str(resposta.content)
    return texto.strip(), ferramentas_usadas


# ---------------------------------------------------------------------------------------
# Wrapper de serving: MLflow ResponsesAgent
# ---------------------------------------------------------------------------------------

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse


class AgenteSuporteResponses(ResponsesAgent):
    """Embrulha o agente na interface ResponsesAgent (para deploy/serving e Agent Evaluation)."""

    def __init__(self, config: ConfigAgente = CONFIG_PADRAO):
        self.config = config

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        pergunta = _ultima_pergunta(request)
        texto, _ = _loop_decisao(self.config, pergunta)
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=texto, id=uuid.uuid4().hex)])


def _ultima_pergunta(request: ResponsesAgentRequest) -> str:
    """Extrai a última mensagem do usuário do request (formato Responses)."""
    for item in reversed(list(getattr(request, "input", []) or [])):
        d = item if isinstance(item, dict) else item.model_dump()
        if d.get("role") == "user":
            conteudo = d.get("content")
            if isinstance(conteudo, str):
                return conteudo
            if isinstance(conteudo, list):
                return " ".join(p.get("text", "") for p in conteudo if isinstance(p, dict))
    return ""


def construir_agente_databricks(config: ConfigAgente = CONFIG_PADRAO) -> AgenteSuporteResponses:
    """Constrói o agente para registro/serving (models-from-code)."""
    return AgenteSuporteResponses(config)


# ---------------------------------------------------------------------------------------
# Interface pública: o pipeline completo (guardrails + agente + observabilidade)
# ---------------------------------------------------------------------------------------

def responder(pergunta: str, config: ConfigAgente = CONFIG_PADRAO) -> Resposta:
    """Executa o pipeline completo e devolve a `Resposta` com a telemetria da requisição."""
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

    # (2) Agente (RAG + tool use), com tracing automático.
    texto, ferramentas = _loop_decisao(config, ge.texto)
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
