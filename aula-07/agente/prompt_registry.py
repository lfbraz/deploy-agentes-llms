"""
prompt_registry.py — o system prompt do agente no MLflow Prompt Registry (Unity Catalog)
=======================================================================================

O prompt não fica no código: ele é um artefato **versionado** no Unity Catalog. Cada versão é
**imutável**; os **aliases** (`champion`, `challenger`) são ponteiros móveis. O agente resolve o
alias a cada requisição, então promover ou fazer rollback é só reapontar o alias:

    registrar_versao(template)  ──►  v1, v2, v3 ...  ◄──  champion / challenger  ◄──  agente
"""

from __future__ import annotations

import mlflow

from .config import AUTOR, ENDPOINT_LLM, PROMPT_UC, SYSTEM_PROMPT_PADRAO


def registrar_versao(template: str, autor: str = AUTOR, mensagem: str = "") -> int:
    """Registra o template como nova versão — ou reaproveita a versão idêntica, se já existir."""
    from mlflow import MlflowClient

    try:
        for v in MlflowClient().search_prompt_versions(PROMPT_UC, max_results=1000):
            if v.template == template:
                return int(v.version)
    except Exception:
        pass   # o prompt ainda não existe: a primeira versão é criada abaixo
    versao = mlflow.genai.register_prompt(
        name=PROMPT_UC, template=template, commit_message=mensagem or f"versão por {autor}",
        tags={"autor": autor, "tested_with": ENDPOINT_LLM})
    return int(versao.version)


def mover_alias(alias: str, versao: int) -> None:
    mlflow.genai.set_prompt_alias(PROMPT_UC, alias=alias, version=versao)


def garantir_prompt_producao() -> int:
    """Setup (exemplo 00): garante que existe um `champion`; senão, registra a 1ª versão produtiva.

    Se já existe um `champion`, não mexemos — quem promove versões é o pipeline (exemplo 06).
    """
    try:
        return int(mlflow.genai.load_prompt(f"prompts:/{PROMPT_UC}@champion").version)
    except Exception:
        versao = registrar_versao(SYSTEM_PROMPT_PADRAO, mensagem="primeira versão produtiva")
        mover_alias("champion", versao)
        return versao
