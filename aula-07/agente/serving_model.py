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
                               FUNC_STATUS_PEDIDO, FUNC_CALCULAR_FRETE, MODELO_UC, PROMPT_UC)

    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_UC}@champion")   # fixado no deploy
    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(
            name="agente_suporte",
            python_model="agente/serving_model.py",          # <- este arquivo
            model_config={"prompt_template": prompt.template, "prompt_version": prompt.version},
            prompts=[prompt],                                 # liga modelo <-> versão do prompt
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

from mlflow.models import ModelConfig, set_model

from agente.agente_suporte import construir_agente_databricks
from agente.config import ConfigAgente

# O container de serving não acessa o Prompt Registry: o exemplo 05 resolve o `champion` no
# deploy e passa a versão (texto + número) no `model_config`. Mudar o prompt em produção =
# mover o alias + novo deploy — e cada versão do modelo fica ligada à versão exata do prompt.
cfg = ModelConfig().to_dict()

# Monta o agente (ChatDatabricks + UC tools + retriever de Vector Search) e o define
# como o modelo do artefato.
set_model(construir_agente_databricks(ConfigAgente(
    prompt_template=cfg["prompt_template"], prompt_version=cfg["prompt_version"])))
