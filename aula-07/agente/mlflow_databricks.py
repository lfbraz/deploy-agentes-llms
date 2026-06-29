"""
mlflow_databricks.py — configuração e pré-requisitos do ambiente
================================================================

Centraliza a integração com o Databricks/MLflow:

  • checagem de pré-requisitos (credenciais + bibliotecas), com mensagem clara quando algo
    falta — em vez de um `ImportError` ou `401` obscuro;
  • configuração do MLflow (tracking no workspace, registry no Unity Catalog);
  • montagem dos nomes de três níveis do Unity Catalog (`catalogo.schema.objeto`).

Dentro de um workspace Databricks, a autenticação vem do próprio runtime; rodando localmente,
use `DATABRICKS_HOST`/`DATABRICKS_TOKEN` (veja o `aula-07/.env.example`).
"""

from __future__ import annotations

import os


def em_databricks() -> bool:
    """Estamos executando dentro de um runtime do Databricks?"""
    return bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))


def credenciais_ok() -> bool:
    """Há como autenticar no Databricks (pelo runtime ou por HOST/TOKEN no ambiente)?"""
    return em_databricks() or bool(os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"))


def exigir_databricks(libs: tuple[str, ...] = ()) -> None:
    """Valida credenciais e bibliotecas; levanta um erro orientativo se faltar algo."""
    if not credenciais_ok():
        raise RuntimeError(
            "Defina DATABRICKS_HOST e DATABRICKS_TOKEN no arquivo aula-07/.env "
            "(use o aula-07/.env.example como base)."
        )
    faltando = [lib for lib in libs if not _tem(lib)]
    if faltando:
        pacotes = " ".join(m.replace("_", "-") for m in faltando)
        raise RuntimeError(
            f"Bibliotecas ausentes: {', '.join(faltando)}. Instale com: pip install {pacotes}"
        )


def _tem(lib: str) -> bool:
    try:
        __import__(lib)
        return True
    except Exception:
        return False


_CONFIGURADO = False


def configurar_mlflow() -> None:
    """Aponta o MLflow para o workspace (tracking) e o Unity Catalog (registry)."""
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


def _catalogo_schema() -> str:
    """Catálogo.schema a partir de MODELO_UC (ex.: 'main.default.agente_suporte' -> 'main.default')."""
    base = os.getenv("MODELO_UC", "main.default.agente_suporte")
    partes = base.split(".")
    return ".".join(partes[:2]) if len(partes) >= 2 else "main.default"


def nome_uc(objeto: str) -> str:
    """Nome de três níveis do Unity Catalog para um objeto (modelo/prompt)."""
    return f"{_catalogo_schema()}.{objeto}"
