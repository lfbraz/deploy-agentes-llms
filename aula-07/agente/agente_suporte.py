"""
agente_suporte.py — o agente embrulhado num MLflow `ResponsesAgent`
===================================================================

Costura tudo num único agente, no caminho que o **Databricks Agent Framework** recomenda: autorar
na interface **`ResponsesAgent`** do MLflow já traz *tracing*, avaliação, deploy e monitoramento
de brinde. É aqui que os pilares das aulas anteriores se encontram:

    PERGUNTA ──►  agente (LLM + tool use: UC Functions + Vector Search RAG)  ──►  RESPOSTA
                                         │
                            observabilidade: mlflow.langchain.autolog()

O "cérebro" é um endpoint Foundation Model (Claude) acessado pelo `ChatDatabricks`. As ferramentas
são as **UC Functions** (`ferramentas.py`) e o **retriever de Vector Search** (`base_vetorial.py`).
O *tool use* não tem mágica: é um **loop explícito** (`bind_tools` → o modelo pede uma ferramenta →
a gente executa → devolve o resultado → repete até ele concluir), igual à Aula 4. E a observabilidade
sai de graça com o `mlflow.langchain.autolog()`: cada ida ao modelo ou a uma ferramenta vira um
*span* dentro da trace.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import mlflow

from .config import CONFIG_PADRAO, LOJA, ConfigAgente
from .mlflow_databricks import configurar_mlflow, exigir_databricks


# ---------------------------------------------------------------------------------------
# A resposta e a telemetria que acompanham cada pergunta.
# ---------------------------------------------------------------------------------------

@dataclass
class Resposta:
    """O que o agente respondeu, junto com o rastro daquela requisição (a 'trace')."""

    pergunta: str
    texto: str
    trace_id: str
    ferramentas_usadas: list[str] = field(default_factory=list)
    latencia_ms: float = 0.0


TELEMETRIA: list[Resposta] = []


def limpar_telemetria() -> None:
    TELEMETRIA.clear()


# ---------------------------------------------------------------------------------------
# O cérebro: ChatDatabricks + as tools (UC Functions + Vector Search) + o loop de decisão.
# ---------------------------------------------------------------------------------------

# Guardamos o LLM já com as ferramentas plugadas, para não remontar tudo a cada pergunta.
_LLMS: dict[str, tuple] = {}


def _construir_llm(config: ConfigAgente):
    """Liga o ChatDatabricks às ferramentas e guarda um de-para nome->ferramenta para o loop."""
    exigir_databricks(libs=("databricks_langchain",))
    configurar_mlflow()
    from databricks_langchain import ChatDatabricks

    from .base_vetorial import retriever_tool
    from .ferramentas import tools_uc

    mlflow.langchain.autolog()      # com isso, modelo e tools já saem rastreados de graça
    llm = ChatDatabricks(endpoint=config.endpoint, temperature=config.temperatura,
                         max_tokens=config.max_tokens)
    tools = list(tools_uc()) + [retriever_tool()]     # as UC Functions mais o retriever do RAG
    return llm.bind_tools(tools), {t.name: t for t in tools}


def _llm(config: ConfigAgente):
    if config.endpoint not in _LLMS:
        _LLMS[config.endpoint] = _construir_llm(config)
    return _LLMS[config.endpoint]


def _loop_decisao(config: ConfigAgente, pergunta: str, max_voltas: int = 5) -> tuple[str, list[str]]:
    """O vaivém do tool use: o modelo pede uma ferramenta, a gente executa e devolve, até ele fechar."""
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
# A embalagem para o serving: o MLflow ResponsesAgent (deploy, serving e Agent Evaluation).
# ---------------------------------------------------------------------------------------

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse


class AgenteSuporteResponses(ResponsesAgent):
    """Veste o agente com a interface ResponsesAgent — o padrão do Databricks Agent Framework."""

    def __init__(self, config: ConfigAgente = CONFIG_PADRAO):
        self.config = config

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        pergunta = _ultima_pergunta(request)
        texto, _ = _loop_decisao(self.config, pergunta)
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=texto, id=uuid.uuid4().hex)])


def _ultima_pergunta(request: ResponsesAgentRequest) -> str:
    """Pesca a última fala do usuário lá no request (que vem no formato Responses)."""
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
    """Entrega o agente prontinho para registrar e servir (models-from-code)."""
    return AgenteSuporteResponses(config)


# ---------------------------------------------------------------------------------------
# A porta de entrada: conversar com o agente e registrar o feedback de quem usou.
# ---------------------------------------------------------------------------------------

def responder(pergunta: str, config: ConfigAgente = CONFIG_PADRAO) -> Resposta:
    """Roda o agente e devolve a `Resposta`, já com o rastro (trace) daquela requisição."""
    t0 = time.perf_counter()
    texto, ferramentas = _loop_decisao(config, pergunta)
    trace_id = mlflow.get_last_active_trace_id() or f"tr-{uuid.uuid4().hex[:12]}"
    resp = Resposta(
        pergunta=pergunta, texto=texto, trace_id=trace_id,
        ferramentas_usadas=ferramentas, latencia_ms=(time.perf_counter() - t0) * 1000,
    )
    TELEMETRIA.append(resp)
    return resp


def registrar_feedback(resp: Resposta, util: bool, comentario: str = "",
                       usuario: str = "cliente@acme.com") -> None:
    """Anexa o 👍/👎 do usuário à trace, como uma *assessment* HUMANA.

    É o **human feedback** do MLflow 3: o veredito de quem usou fica grudado na trace e vira
    combustível para a avaliação e para montar *golden sets* (veja a Aula 6).
    """
    from mlflow.entities import AssessmentSource

    mlflow.log_feedback(
        trace_id=resp.trace_id,
        name="user_feedback",
        value=bool(util),
        source=AssessmentSource(source_type="HUMAN", source_id=usuario),
        rationale=comentario or None,
    )
