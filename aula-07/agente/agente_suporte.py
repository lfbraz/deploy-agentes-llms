"""
agente_suporte.py — o agente empacotado num MLflow `ResponsesAgent`
===================================================================

Desenvolvido como um único agente, no caminho que o **Databricks Agent Framework** recomenda: autorar
na interface **`ResponsesAgent`** do MLflow já traz *tracing*, avaliação, deploy e monitoramento integrados. É aqui que os pilares das aulas anteriores se encontram:

    PERGUNTA ──►  agente (LLM + tool use: UC Functions + Vector Search RAG)  ──►  RESPOSTA
                                         │
                            observabilidade: mlflow.langchain.autolog()

O "cérebro" é um endpoint Foundation Model acessado pelo `ChatDatabricks`. As ferramentas
são as **UC Functions** (`ferramentas.py`) e o **retriever de Vector Search** (`base_vetorial.py`).
O *tool use* é um **loop explícito** (`bind_tools` → o modelo pede uma ferramenta → executamos →
devolvemos o resultado → repete até ele concluir). E a observabilidade não exige esforço extra: com o `mlflow.langchain.autolog()`, cada chamada ao modelo ou a uma ferramenta vira um *span* dentro da trace.

O system prompt vem do **Prompt Registry** (alias `champion`), carregado dentro da trace: cada
resposta fica ligada à versão do prompt que a gerou.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import mlflow
from mlflow.entities import SpanType

from .config import LOJA, ConfigAgente
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
    prompt_versao: int | None = None


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

    mlflow.langchain.autolog()      # a partir daqui, modelo e tools são rastreados automaticamente
    llm = ChatDatabricks(endpoint=config.endpoint, temperature=config.temperatura,
                         max_tokens=config.max_tokens)
    tools = list(tools_uc()) + [retriever_tool()]     # as UC Functions mais o retriever do RAG
    return llm.bind_tools(tools), {t.name: t for t in tools}


def _llm(config: ConfigAgente):
    if config.endpoint not in _LLMS:
        _LLMS[config.endpoint] = _construir_llm(config)
    return _LLMS[config.endpoint]


def _loop_decisao(config: ConfigAgente, system_prompt: str, pergunta: str,
                  max_voltas: int = 5) -> tuple[str, list[str]]:
    """Loop do tool use: o modelo pede uma ferramenta, executamos e devolvemos, até ele concluir."""
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    llm_tools, registro = _llm(config)
    mensagens = [SystemMessage(system_prompt), HumanMessage(pergunta)]
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


@mlflow.trace(name="agente_suporte", span_type=SpanType.AGENT)
def _executar(config: ConfigAgente, pergunta: str) -> tuple[str, list[str], int, str]:
    """Uma requisição = uma trace: carrega o prompt pelo alias, roda o loop e etiqueta a trace."""
    try:
        prompt = mlflow.genai.load_prompt(config.prompt_uri)   # dentro da trace -> fica vinculado
    except Exception as e:
        raise RuntimeError(f"Não consegui carregar {config.prompt_uri}. Rode o Passo 0 "
                           "(00_setup_recursos) para registrar a versão produtiva.") from e
    versao = prompt.version
    system_prompt = prompt.format(loja=LOJA)
    if mlflow.get_current_active_span():   # sem trace ativa (ex.: validação do log_model), pula
        mlflow.update_current_trace(tags={
            "prompt_uri": config.prompt_uri,
            "prompt_version": str(versao),
            "config_hash": config.hash(system_prompt),
        })
    texto, ferramentas = _loop_decisao(config, system_prompt, pergunta)
    return texto, ferramentas, int(versao), mlflow.get_active_trace_id()


# ---------------------------------------------------------------------------------------
# A embalagem para o serving: o MLflow ResponsesAgent (deploy, serving e Agent Evaluation).
# ---------------------------------------------------------------------------------------

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse


class AgenteSuporteResponses(ResponsesAgent):
    """Expõe o agente pela interface ResponsesAgent — o padrão do Databricks Agent Framework."""

    def __init__(self, config: ConfigAgente | None = None):
        self.config = config or ConfigAgente()

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        configurar_mlflow()
        pergunta = _ultima_pergunta(request)
        texto, ferramentas, versao, trace_id = _executar(self.config, pergunta)
        # O trace_id volta para quem chamou: é com ele que a app grava o 👍/👎 (registrar_feedback).
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=texto, id=uuid.uuid4().hex)],
            custom_outputs={"trace_id": trace_id, "prompt_version": versao,
                            "ferramentas": ferramentas})


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


def construir_agente_databricks(config: ConfigAgente | None = None) -> AgenteSuporteResponses:
    """Constrói o agente pronto para registrar e servir (models-from-code)."""
    return AgenteSuporteResponses(config)


# ---------------------------------------------------------------------------------------
# A porta de entrada: conversar com o agente e registrar o feedback de quem usou.
# ---------------------------------------------------------------------------------------

def responder(pergunta: str, config: ConfigAgente | None = None) -> Resposta:
    """Roda o agente e devolve a `Resposta`, já com o rastro (trace) daquela requisição."""
    configurar_mlflow()   # antes da trace: tracking/registry/experimento no lugar certo
    t0 = time.perf_counter()
    texto, ferramentas, versao, trace_id = _executar(config or ConfigAgente(), pergunta)
    resp = Resposta(
        pergunta=pergunta, texto=texto, trace_id=trace_id, ferramentas_usadas=ferramentas,
        latencia_ms=(time.perf_counter() - t0) * 1000, prompt_versao=versao,
    )
    TELEMETRIA.append(resp)
    return resp


def registrar_feedback(resp: Resposta | str, util: bool, comentario: str = "",
                       usuario: str = "cliente@acme.com") -> None:
    """Anexa o 👍/👎 do usuário à trace, como uma *assessment* HUMANA.

    É o **human feedback** do MLflow 3: o veredito do usuário fica anexado à trace e alimenta
    a avaliação e a curadoria de *golden sets* (veja a Aula 6). Aceita a `Resposta` local ou só
    o `trace_id` — que é o que o endpoint servido devolve em `custom_outputs`.
    """
    from mlflow.entities import AssessmentSource

    # As traces são exportadas de forma assíncrona: esvaziamos a fila local e, se a trace ainda
    # não chegou ao servidor (comum logo após a resposta do endpoint), tentamos de novo.
    mlflow.flush_trace_async_logging()
    for tentativa in range(6):
        try:
            mlflow.log_feedback(
                trace_id=resp if isinstance(resp, str) else resp.trace_id,
                name="user_feedback",
                value=bool(util),
                source=AssessmentSource(source_type="HUMAN", source_id=usuario),
                rationale=comentario or None,
            )
            return
        except Exception as e:
            if "not found" not in str(e).lower() or tentativa == 5:
                raise
            time.sleep(5)
