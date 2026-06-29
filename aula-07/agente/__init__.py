"""
Pacote `agente` — o agente de suporte como uma SOLUÇÃO DE SOFTWARE COMPLETA.

Esta é a peça central da Aula 07: antes de falar de versionamento, CI/CD, serving e
monitoramento, precisamos de um agente *de verdade* para operar. Ele reúne os pilares das
aulas anteriores:

    config.py        -> a "tripla" versionada: system_prompt + modelo (endpoint) + geração
    base_vetorial.py -> RAG: base de conhecimento (políticas/FAQ) num índice vetorial
    ferramentas.py   -> tool use: o agente consulta pedido, calcula frete e busca política
    guardrails.py    -> segurança: filtros de ENTRADA (jailbreak/PII) e SAÍDA (vazamento)
    agente_suporte.py-> orquestra tudo num MLflow `ResponsesAgent` (padrão Databricks)

Tudo é importável e executável **offline** (só `mlflow` + `numpy` + stdlib). As bibliotecas
do Databricks (`databricks-langchain`, `langgraph`) são carregadas de forma PREGUIÇOSA — só
quando há `DATABRICKS_HOST`/`DATABRICKS_TOKEN` e o agente "vivo" é solicitado. Sem credenciais,
o agente roda num **cérebro determinístico de referência**, suficiente para demonstrar a
mecânica e para servir de baseline aos testes de regressão (a regra §5 do curso).
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
