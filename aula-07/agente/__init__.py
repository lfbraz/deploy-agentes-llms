"""
Pacote `agente` — o agente de suporte como uma solução de software completa.

Este pacote é a peça central da Aula 07: antes de falar de versionamento, CI/CD, serving e
monitoramento, precisamos do agente que será operado. Ele é construído baseado no
**Databricks Agent Framework**:

    config.py        -> configuração central: a "tripla" versionada + nomes UC + endpoints
    prompt_registry.py -> o system prompt versionado no Prompt Registry (aliases champion/challenger)
    ferramentas.py   -> tool use: UC Functions (status via tabela `pedidos`, frete) + UCFunctionToolkit
    base_vetorial.py -> RAG: índice de Vector Search (políticas) + VectorSearchRetrieverTool
    agente_suporte.py-> orquestra tudo num MLflow `ResponsesAgent` + mlflow.langchain.autolog
    serving_model.py -> entrypoint models-from-code para log_model / agents.deploy

A autenticação vem do runtime do Databricks ou, na sua máquina, do
`DATABRICKS_HOST`/`DATABRICKS_TOKEN` (veja o `aula-07/.env.example`). Os guardrails de produção
você liga na própria UI do workspace (AI Gateway / Guardrails); o comportamento do agente é
validado pelo **gate de aprovação** (exemplo 04) — testes unitários sobre situações conhecidas,
com o status do pedido vindo de uma tabela real do Unity Catalog.
"""

# Antes de ler o ambiente, carregamos o aula-07/.env (HOST/TOKEN, catálogo/schema, endpoints).
try:
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from .config import ConfigAgente
from .agente_suporte import Resposta, registrar_feedback, responder

__all__ = ["ConfigAgente", "Resposta", "responder", "registrar_feedback"]
