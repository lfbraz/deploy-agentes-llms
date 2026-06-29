"""
mlflow_databricks.py — configuração e pré-requisitos do ambiente Databricks
===========================================================================

A Aula 07 roda **sempre contra serviços reais** do Databricks/MLflow (não há modo offline).
Este módulo centraliza:

  • a checagem de pré-requisitos (`DATABRICKS_HOST`/`DATABRICKS_TOKEN` + libs), com mensagem
    de erro clara quando algo falta — em vez de um `ImportError` ou `401` obscuro;
  • a configuração do MLflow para apontar o *tracking* para o workspace e o *registry* para o
    **Unity Catalog**;
  • a montagem dos nomes de três níveis do Unity Catalog (`catalogo.schema.objeto`).

Os módulos do pacote continuam IMPORTÁVEIS sem credenciais (os imports pesados são preguiçosos);
a exigência só dispara quando você executa algo que fala com o Databricks.
"""

from __future__ import annotations

import os


def credenciais_ok() -> bool:
    """Há credenciais do Databricks no ambiente (.env)?"""
    return bool(os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"))


def exigir_databricks(libs: tuple[str, ...] = ()) -> None:
    """Valida credenciais e bibliotecas; levanta erro didático se faltar algo."""
    if not credenciais_ok():
        raise RuntimeError(
            "Esta aula roda contra o Databricks (sem modo offline). Defina DATABRICKS_HOST e "
            "DATABRICKS_TOKEN no arquivo aula-07/.env (use o aula-07/.env.example como base)."
        )
    faltando = []
    for lib in libs:
        try:
            __import__(lib)
        except Exception:
            faltando.append(lib)
    if faltando:
        pacotes = " ".join(m.replace("_", "-") for m in faltando)
        raise RuntimeError(
            f"Bibliotecas ausentes: {', '.join(faltando)}. Instale com: pip install {pacotes}"
        )


_CONFIGURADO = False


def configurar_mlflow() -> None:
    """Aponta o MLflow para o workspace (tracking) e o Unity Catalog (registry)."""
    global _CONFIGURADO
    if _CONFIGURADO:
        return
    exigir_databricks(libs=("mlflow",))
    import mlflow

    # Em notebooks do Databricks o tracking já é o workspace; localmente, "databricks" usa
    # DATABRICKS_HOST/TOKEN do ambiente.
    if not os.getenv("DATABRICKS_RUNTIME_VERSION"):
        mlflow.set_tracking_uri("databricks")
    try:
        mlflow.set_registry_uri("databricks-uc")     # Model/Prompt Registry no Unity Catalog
    except Exception:
        pass
    _CONFIGURADO = True


def _catalogo_schema() -> str:
    """Catálogo.schema a partir de MODELO_UC (ex.: 'main.default.agente_suporte' -> 'main.default')."""
    base = os.getenv("MODELO_UC", "main.default.agente_suporte")
    partes = base.split(".")
    return ".".join(partes[:2]) if len(partes) >= 2 else "main.default"


def nome_uc(objeto: str) -> str:
    """Nome de três níveis do Unity Catalog para um objeto (modelo/prompt). Ex.: agente_suporte."""
    return f"{_catalogo_schema()}.{objeto}"
