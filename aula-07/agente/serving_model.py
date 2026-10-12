"""
serving_model.py — entrypoint "models-from-code" do MLflow para o agente
========================================================================

Este é o arquivo que o MLflow executa ao registrar o agente pela abordagem
*models-from-code*. O único requisito: ao executá-lo, o MLflow espera encontrar um
`set_model(...)` apontando para a instância do agente.

No registro, listamos os **recursos governados** que o agente usa (o exemplo 05 traz o
`log_model` completo) — é o que permite ao serving herdar as credenciais automaticamente, via
*automatic authentication*:

    import mlflow
    from mlflow.models.resources import (
        DatabricksServingEndpoint, DatabricksVectorSearchIndex, DatabricksFunction)
    from agente.config import (ENDPOINT_LLM, ENDPOINT_EMBEDDINGS, INDICE_POLITICAS,
                               FUNC_STATUS_PEDIDO, FUNC_CALCULAR_FRETE, MODELO_UC)

    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(
            name="agente_suporte",
            python_model="agente/serving_model.py",          # <- este arquivo
            resources=[
                DatabricksServingEndpoint(endpoint_name=ENDPOINT_LLM),
                DatabricksServingEndpoint(endpoint_name=ENDPOINT_EMBEDDINGS),
                DatabricksVectorSearchIndex(index_name=INDICE_POLITICAS),
                DatabricksFunction(function_name=FUNC_STATUS_PEDIDO),
                DatabricksFunction(function_name=FUNC_CALCULAR_FRETE),
            ],
            registered_model_name=MODELO_UC,                  # Unity Catalog
            pip_requirements=["mlflow", "databricks-langchain", "databricks-vectorsearch",
                              "unitycatalog-ai"],
        )

    from databricks import agents
    agents.deploy(MODELO_UC, info.registered_model_version)   # Model Serving
"""

from mlflow.models import set_model

from agente.agente_suporte import construir_agente_databricks

# Monta o agente (ChatDatabricks + UC tools + retriever de Vector Search) e o define como o
# modelo do artefato. O system prompt é lido do Prompt Registry (`champion`) a cada requisição —
# o endpoint roda como o Service Principal do exemplo 00b, que tem acesso ao prompt.
set_model(construir_agente_databricks())
