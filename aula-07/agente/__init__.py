"""
Pacote `agente` — o agente de suporte como uma SOLUÇÃO DE SOFTWARE COMPLETA.

Esta é a peça central da Aula 07: antes de falar de versionamento, CI/CD, serving e
monitoramento, precisamos do agente que será operado. Ele reúne os pilares das aulas
anteriores:

    config.py        -> a "tripla" versionada: system_prompt + modelo (endpoint) + geração
    base_vetorial.py -> RAG: base de conhecimento (políticas/FAQ) num índice vetorial
    ferramentas.py   -> tool use: o agente consulta pedido, calcula frete e busca política
    guardrails.py    -> segurança: filtros de ENTRADA (jailbreak/PII) e SAÍDA (vazamento)
    agente_suporte.py-> orquestra tudo num MLflow `ResponsesAgent` (padrão Databricks)

A biblioteca `databricks-langchain` é carregada de forma preguiçosa, ao construir o agente. A
autenticação vem do runtime do Databricks ou, localmente, de `DATABRICKS_HOST`/`DATABRICKS_TOKEN`
(veja o `aula-07/.env.example`).
"""

# Carrega o aula-07/.env (DATABRICKS_HOST/TOKEN, endpoints) antes de ler o ambiente.
try:
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from .config import CONFIG_PADRAO, ConfigAgente
from .agente_suporte import Resposta, responder

__all__ = ["CONFIG_PADRAO", "ConfigAgente", "Resposta", "responder"]
