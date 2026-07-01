"""
Pacote `agente` — o agente de suporte como uma solução de software completa.

Este pacote é a peça central da Aula 07: antes de falar de versionamento, CI/CD, serving e
monitoramento, precisamos do agente que será operado. Ele é construído baseado no Databricks Agent Framework**:

    config.py        -> configuração central: a "tripla" versionada + nomes UC + endpoints
    ferramentas.py   -> tool use: UC Functions (status do pedido, frete) via UCFunctionToolkit
    base_vetorial.py -> RAG: índice de Vector Search (políticas) + VectorSearchRetrieverTool
    agente_suporte.py-> orquestra tudo num MLflow `ResponsesAgent` + mlflow.langchain.autolog
    serving_model.py -> entrypoint models-from-code para log_model / agents.deploy

A autenticação vem do runtime do Databricks ou, na sua máquina, do
`DATABRICKS_HOST`/`DATABRICKS_TOKEN` (veja o `aula-07/.env.example`). Os guardrails de produção
você liga na própria UI do workspace (AI Gateway / Guardrails); por aqui, um teste de *jailbreak*
que precisa ser recusado entra como caso do **gate de aprovação** (exemplo 04).
"""

# Antes de ler o ambiente, carregamos o aula-07/.env (HOST/TOKEN, catálogo/schema, endpoints).
try:
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from .config import CONFIG_PADRAO, ConfigAgente
from .agente_suporte import Resposta, registrar_feedback, responder

__all__ = ["CONFIG_PADRAO", "ConfigAgente", "Resposta", "responder", "registrar_feedback"]
