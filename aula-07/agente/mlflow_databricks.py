"""
mlflow_databricks.py — a conversa do agente com o Databricks/MLflow
===================================================================

Este é o módulo que arruma a casa antes de qualquer coisa rodar. Ele faz duas tarefas chatas
porém essenciais: confere se temos credencial e as bibliotecas na mão (e, quando falta algo,
avisa em bom português em vez de deixar estourar um `ImportError` ou um `401` enigmático) e
aponta o MLflow para os lugares certos — o tracking no workspace e o registry no Unity Catalog.

Dentro de um workspace Databricks a autenticação já vem de graça pelo runtime; rodando na sua
máquina, basta o `DATABRICKS_HOST`/`DATABRICKS_TOKEN` (veja o `aula-07/.env.example`). Os nomes
de três níveis do Unity Catalog moram no `config.py`.
"""

from __future__ import annotations

import os

from .config import uc  # nomes de três níveis do Unity Catalog


def em_databricks() -> bool:
    """Estamos executando dentro de um runtime do Databricks?"""
    return bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))


def credenciais_ok() -> bool:
    """Há como autenticar no Databricks (pelo runtime ou por HOST/TOKEN no ambiente)?"""
    return em_databricks() or bool(os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"))


def exigir_databricks(libs: tuple[str, ...] = ()) -> None:
    """Confere credenciais e bibliotecas e, se faltar algo, diz exatamente o que fazer."""
    if not credenciais_ok():
        raise RuntimeError(
            "Defina DATABRICKS_HOST e DATABRICKS_TOKEN no arquivo aula-07/.env "
            "(use o aula-07/.env.example como base)."
        )
    faltando = [lib for lib in libs if not _tem(lib)]
    if faltando:
        pacotes = " ".join(_PACOTE_PIP.get(m, m.replace("_", "-")) for m in faltando)
        raise RuntimeError(
            f"Bibliotecas ausentes: {', '.join(faltando)}. Instale com: pip install {pacotes}"
        )


# Nem sempre o nome do import bate com o do pacote pip — este de-para deixa a dica de erro certa.
_PACOTE_PIP = {
    "databricks_langchain": "databricks-langchain",
    "databricks.vector_search": "databricks-vectorsearch",
    "unitycatalog": "unitycatalog-ai",
}


def _tem(lib: str) -> bool:
    try:
        __import__(lib)
        return True
    except Exception:
        return False


_CONFIGURADO = False


def configurar_mlflow() -> None:
    """Aponta o MLflow para os lugares certos: tracking no workspace, registry no Unity Catalog."""
    global _CONFIGURADO
    if _CONFIGURADO:
        return
    exigir_databricks(libs=("mlflow",))
    import mlflow

    if not em_databricks():
        mlflow.set_tracking_uri("databricks")
    try:
        mlflow.set_registry_uri("databricks-uc")
    except Exception:
        pass
    _CONFIGURADO = True


def nome_uc(objeto: str) -> str:
    """Nome de três níveis do Unity Catalog para um objeto (compatibilidade: ver config.uc)."""
    return uc(objeto)
