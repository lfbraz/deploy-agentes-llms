# Databricks notebook source
# MAGIC %md
# MAGIC # Passo 5 — Servir o agente: log_model, Unity Catalog e Model Serving
# MAGIC
# MAGIC Aprovado no gate, o agente é registrado e implantado num **endpoint de Model Serving**. No
# MAGIC Databricks Agent Framework isso são três movimentos:
# MAGIC
# MAGIC 1. **`log_model`** (models-from-code) declarando os **recursos governados** que o agente usa
# MAGIC    (endpoints, índice de Vector Search, UC Functions) — para o serving herdar as credenciais
# MAGIC    via *automatic authentication*;
# MAGIC 2. **registrar no Unity Catalog** (modelo com nome de três níveis, versões imutáveis);
# MAGIC 3. **`agents.deploy`** — provisiona o endpoint de Model Serving do agente.
# MAGIC
# MAGIC **Pré-requisitos:** Passo 0 (recursos) e Passo 4 (gate verde). O deploy leva alguns minutos.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ambiente

# COMMAND ----------

# MAGIC %pip install -U mlflow databricks-langchain databricks-vectorsearch unitycatalog-ai databricks-agents

# COMMAND ----------

try:
    dbutils.library.restartPython()  # noqa: F821
except NameError:
    pass

# COMMAND ----------

import os
import sys

try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()
while _dir != os.path.dirname(_dir):
    if os.path.isdir(os.path.join(_dir, "agente")):
        BASE = _dir
        sys.path.insert(0, _dir)
        break
    _dir = os.path.dirname(_dir)

import mlflow

from agente.config import (
    ENDPOINT_EMBEDDINGS, ENDPOINT_LLM, FUNC_CALCULAR_FRETE, FUNC_STATUS_PEDIDO,
    INDICE_POLITICAS, MODELO_UC,
)
from agente.mlflow_databricks import configurar_mlflow, credenciais_ok

if not credenciais_ok():
    raise SystemExit("Configure DATABRICKS_HOST/TOKEN no aula-07/.env (veja o .env.example).")

configurar_mlflow()
print("modelo (Unity Catalog):", MODELO_UC)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Registrar o agente declarando seus recursos
# MAGIC
# MAGIC Os `resources` listam tudo que o agente consome em tempo de execução. O serving usa essa lista
# MAGIC para conceder as credenciais certas automaticamente (sem PAT embutido no código).

# COMMAND ----------

from mlflow.models.resources import (
    DatabricksFunction, DatabricksServingEndpoint, DatabricksVectorSearchIndex,
)

input_example = {"input": [{"role": "user", "content": "Qual a garantia dos eletrônicos?"}]}

with mlflow.start_run():
    info = mlflow.pyfunc.log_model(
        name="agente_suporte",
        python_model=os.path.join(BASE, "agente", "serving_model.py"),  # models-from-code
        code_paths=[os.path.join(BASE, "agente")],                      # empacota o pacote
        input_example=input_example,
        resources=[
            DatabricksServingEndpoint(endpoint_name=ENDPOINT_LLM),
            DatabricksServingEndpoint(endpoint_name=ENDPOINT_EMBEDDINGS),
            DatabricksVectorSearchIndex(index_name=INDICE_POLITICAS),
            DatabricksFunction(function_name=FUNC_STATUS_PEDIDO),
            DatabricksFunction(function_name=FUNC_CALCULAR_FRETE),
        ],
        registered_model_name=MODELO_UC,     # registra direto no Unity Catalog
        pip_requirements=["mlflow", "databricks-langchain", "databricks-vectorsearch",
                          "unitycatalog-ai"],
    )

print("registrado:", MODELO_UC, "v", info.registered_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Validar antes do deploy
# MAGIC
# MAGIC `mlflow.models.predict` roda o modelo num subprocesso isolado, com o ambiente declarado — pega
# MAGIC erros de dependência/recurso antes de provisionar o endpoint.

# COMMAND ----------

# (Opcional) Validação pré-deploy DESATIVADA — descomente para ligá-la.
# `env_manager="virtualenv"` valida o modelo num ambiente ISOLADO, recriado a partir do
# pip_requirements — assim pegamos dependência faltando antes do deploy. Usamos virtualenv porque
# o `uv` não está no cluster; "local" validaria no ambiente atual, sem isolar.
# mlflow.models.predict(
#     model_uri=info.model_uri,
#     input_data=input_example,
#     env_manager="virtualenv",
# )
print("validação pré-deploy pulada (célula comentada).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Deploy para Model Serving
# MAGIC
# MAGIC `agents.deploy` provisiona o endpoint do agente (com a *review app* e a tabela de inferência
# MAGIC para feedback/monitoramento). Leva alguns minutos até ficar `READY`.

# COMMAND ----------

from databricks import agents

deployment = agents.deploy(MODELO_UC, info.registered_model_version, scale_to_zero=True)
print("endpoint   :", deployment.endpoint_name)
print("query URL  :", deployment.query_endpoint)
print("review app :", getattr(deployment, "review_app_url", "—"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Consultar o endpoint (quando estiver READY)
# MAGIC
# MAGIC Depois que o endpoint sobe, consulte-o pelo deploy client. (Logo após o deploy ele ainda pode
# MAGIC estar provisionando — rode esta célula novamente em alguns minutos, ou teste no AI Playground.)

# COMMAND ----------

try:
    cliente = mlflow.deployments.get_deploy_client("databricks")
    resp = cliente.predict(endpoint=deployment.endpoint_name, inputs=input_example)
    print(resp)
except Exception as e:
    print("endpoint ainda provisionando — tente de novo em alguns minutos.\n", e)

# COMMAND ----------

# MAGIC %md
# MAGIC O agente está **servido** e governado: versão imutável no Unity Catalog, recursos declarados,
# MAGIC credenciais automáticas e endpoint com *review app*. O Passo 6 amarra tudo numa esteira de CI/CD.
