"""
mlflow_databricks.py — a conversa do agente com o Databricks/MLflow
===================================================================

Módulo de preparação do ambiente, executado antes de qualquer outra coisa. Ele faz duas tarefas
essenciais: confere se há credenciais e bibliotecas disponíveis (e, quando falta algo, mostra uma
mensagem de erro clara em vez de deixar estourar um `ImportError` ou um `401` genérico) e
aponta o MLflow para os lugares certos — o tracking no workspace e o registry no Unity Catalog.

Dentro de um workspace Databricks a autenticação vem do próprio runtime; rodando na sua
máquina, basta o `DATABRICKS_HOST`/`DATABRICKS_TOKEN` (veja o `aula-07/.env.example`). Os nomes
de três níveis do Unity Catalog ficam no `config.py`.
"""

from __future__ import annotations

import os

from .config import uc  # nomes de três níveis do Unity Catalog


def em_databricks() -> bool:
    """Estamos executando dentro de um runtime do Databricks (notebook/job)?"""
    return bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))


def em_model_serving() -> bool:
    """Estamos dentro de um endpoint de Model Serving?

    Lá NÃO existem DATABRICKS_HOST/TOKEN: a autenticação é automática, concedida a partir dos
    `resources` declarados no `log_model`. Por isso não podemos exigir credenciais neste contexto.
    """
    return bool(
        os.getenv("IS_IN_DB_MODEL_SERVING_ENV")
        or os.getenv("DB_MODEL_SERVING_HOST_URL")
        or os.getenv("MODEL_SERVING_ENDPOINT_NAME")
    )


def credenciais_ok() -> bool:
    """Há como autenticar no Databricks (runtime, Model Serving ou HOST/TOKEN no ambiente)?"""
    return (
        em_databricks()
        or em_model_serving()
        or bool(os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"))
    )


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

    if em_model_serving():
        # No serving, o endpoint já cuida de auth e tracing — não reconfiguramos tracking/registry
        # nem experimento (não há workspace-path aqui, e mexer nisso só atrapalharia).
        _CONFIGURADO = True
        return

    if not em_databricks():
        mlflow.set_tracking_uri("databricks")
    try:
        mlflow.set_registry_uri("databricks-uc")
    except Exception:
        pass
    _configurar_experimento(mlflow)
    _CONFIGURADO = True


def _caminho_experimento() -> str | None:
    """O caminho do experimento: usa MLFLOW_EXPERIMENT ou monta /Users/<você>/experimento_aula_07."""
    from .config import MLFLOW_EXPERIMENT

    if MLFLOW_EXPERIMENT:
        return MLFLOW_EXPERIMENT
    try:
        from databricks.sdk import WorkspaceClient

        eu = WorkspaceClient().current_user.me().user_name
        return f"/Users/{eu}/experimento_aula_07"
    except Exception:
        return None


def _configurar_experimento(mlflow) -> None:
    """Escolhe o experimento e, quando dá, manda os traces para tabelas no Unity Catalog.

    Gravar traces no UC precisa de um SQL warehouse (MLFLOW_TRACING_SQL_WAREHOUSE_ID) e do preview
    "OpenTelemetry on Databricks" ligado. Se algo faltar, caímos num experimento comum de workspace
    — sem travar o resto do notebook.
    """
    from .config import CATALOGO, SCHEMA, SQL_WAREHOUSE_ID, TRACE_TABLE_PREFIX

    nome = _caminho_experimento()
    if not nome:
        return

    if SQL_WAREHOUSE_ID:
        try:
            os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID
            from mlflow.entities.trace_location import UnityCatalog

            mlflow.set_experiment(
                experiment_name=nome,
                trace_location=UnityCatalog(catalog_name=CATALOGO, schema_name=SCHEMA,
                                            table_prefix=TRACE_TABLE_PREFIX),
            )
            print(f"traces indo para o Unity Catalog: {CATALOGO}.{SCHEMA}.{TRACE_TABLE_PREFIX}_*")
            return
        except Exception as e:
            print("aviso: não deu para vincular os traces ao Unity Catalog (precisa do preview "
                  f"'OpenTelemetry on Databricks' + SQL warehouse). Usando experimento comum. {e}")

    try:
        mlflow.set_experiment(nome)
    except Exception as e:
        print(f"aviso: não consegui configurar o experimento '{nome}': {e}")


def nome_uc(objeto: str) -> str:
    """Nome de três níveis do Unity Catalog para um objeto (compatibilidade: ver config.uc)."""
    return uc(objeto)
