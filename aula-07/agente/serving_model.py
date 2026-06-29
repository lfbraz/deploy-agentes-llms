"""
serving_model.py — entrypoint "models-from-code" do MLflow para o agente
========================================================================

Este arquivo NÃO é importado pelos exemplos offline. Ele é o artefato que o MLflow executa
ao registrar o agente com a abordagem *models-from-code*:

    import mlflow
    from mlflow.models.resources import DatabricksServingEndpoint
    from agente.config import CONFIG_PADRAO

    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(
            name="agente-suporte",
            python_model="agente/serving_model.py",          # <- este arquivo
            resources=[DatabricksServingEndpoint(endpoint_name=CONFIG_PADRAO.endpoint)],
            registered_model_name="main.default.agente_suporte",   # Unity Catalog
            pip_requirements=["mlflow", "databricks-langchain", "langgraph"],
        )

Ao executar este arquivo, o MLflow espera que `set_model(...)` seja chamado com a instância
do agente. Em seguida, o deploy para Model Serving é feito com:

    from databricks import agents
    agents.deploy("main.default.agente_suporte", info.registered_model_version)
"""

from mlflow.models import set_model

from agente.agente_suporte import construir_agente_databricks
from agente.config import CONFIG_PADRAO

# Constrói o agente vivo (ChatDatabricks + LangGraph) e o registra como o modelo do artefato.
set_model(construir_agente_databricks(CONFIG_PADRAO))
