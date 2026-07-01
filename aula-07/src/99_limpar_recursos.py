# Databricks notebook source
# MAGIC %md
# MAGIC # (Opcional) Limpeza — remover a estrutura da aula no Unity Catalog
# MAGIC
# MAGIC Apaga os recursos que a Aula 07 cria, no **catálogo/schema em uso** (o que vem do `.env` —
# MAGIC `UC_CATALOG`/`UC_SCHEMA`, padrão `aula_07.default`):
# MAGIC
# MAGIC - o **endpoint de serving** do agente (criado pelo `agents.deploy`);
# MAGIC - o **modelo registrado** e o **prompt** do Prompt Registry;
# MAGIC - o **índice** e o **endpoint de Vector Search**;
# MAGIC - as **tabelas** (`pedidos`, `politicas`) e as **UC Functions**.
# MAGIC
# MAGIC > ⚠️ **É destrutivo: zera o ambiente ATUAL da aula.** Rode quando quiser recomeçar do zero —
# MAGIC > o `00_setup_recursos` recria tudo (e o Passo 5 refaz o deploy). O experimento do MLflow e as
# MAGIC > tabelas de traces (`experimento_aula_07_*`) são preservados como histórico; o catálogo/schema
# MAGIC > em si também fica (há um DROP opcional comentado no final).

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
        sys.path.insert(0, _dir)
        break
    _dir = os.path.dirname(_dir)

from agente.config import (
    CATALOGO, ENDPOINT_VECTOR_SEARCH, FUNC_CALCULAR_FRETE, FUNC_STATUS_PEDIDO,
    INDICE_POLITICAS, MODELO_UC, PROMPT_UC, SCHEMA, TABELA_PEDIDOS, TABELA_POLITICAS,
)
from agente.mlflow_databricks import configurar_mlflow, credenciais_ok

if not credenciais_ok():
    raise SystemExit("Configure DATABRICKS_HOST/TOKEN no aula-07/.env (veja o .env.example).")

configurar_mlflow()

print(f"vou remover a estrutura da aula em {CATALOGO}.{SCHEMA}:")
for x in [MODELO_UC, PROMPT_UC, INDICE_POLITICAS, f"endpoint VS: {ENDPOINT_VECTOR_SEARCH}",
          TABELA_PEDIDOS, TABELA_POLITICAS, FUNC_STATUS_PEDIDO, FUNC_CALCULAR_FRETE]:
    print("   -", x)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Endpoint de serving do agente
# MAGIC O deploy (Passo 5) depende do modelo registrado, então o endpoint sai primeiro.

# COMMAND ----------

try:
    from databricks import agents

    agents.delete_deployment(MODELO_UC)
    print(f">> deployment removido: {MODELO_UC}")
except Exception as e:
    print(f"(serving) nada a remover ou sem permissão: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Modelo registrado e prompt
# MAGIC Ambos são objetos do Unity Catalog — apagamos pelo `MlflowClient`. Para o prompt, é preciso
# MAGIC apagar as versões antes do objeto.

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

try:
    client.delete_registered_model(MODELO_UC)
    print(f">> modelo removido: {MODELO_UC}")
except Exception as e:
    print(f"(modelo) nada a remover ou sem permissão: {e}")

try:
    # O UC exige remover as versões antes do prompt em si.
    for v in range(1, 50):
        try:
            client.delete_prompt_version(PROMPT_UC, str(v))
        except Exception:
            pass
    client.delete_prompt(PROMPT_UC)
    print(f">> prompt removido: {PROMPT_UC}")
except Exception as e:
    print(f"(prompt) nada a remover, versões restantes ou sem permissão: {e}\n"
          f"        se necessário, apague pelo Catalog Explorer em {CATALOGO}.{SCHEMA}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Índice e endpoint de Vector Search
# MAGIC O índice não é apagado por SQL — usamos o `VectorSearchClient`. O endpoint (o compute) também
# MAGIC é exclusivo da aula, então sai junto.

# COMMAND ----------

try:
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient(disable_notice=True)
    try:
        vsc.delete_index(index_name=INDICE_POLITICAS)
    except TypeError:
        vsc.delete_index(endpoint_name=ENDPOINT_VECTOR_SEARCH, index_name=INDICE_POLITICAS)
    print(f">> índice removido: {INDICE_POLITICAS}")
except Exception as e:
    print(f"(índice) nada a remover ou sem permissão: {e}")

try:
    vsc.delete_endpoint(name=ENDPOINT_VECTOR_SEARCH)
    print(f">> endpoint de Vector Search removido: {ENDPOINT_VECTOR_SEARCH}")
except Exception as e:
    print(f"(endpoint VS) nada a remover ou sem permissão: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Tabelas e UC Functions
# MAGIC Aqui um `DROP ... IF EXISTS` resolve (via `spark`).

# COMMAND ----------

try:
    for tabela in (TABELA_PEDIDOS, TABELA_POLITICAS):
        spark.sql(f"DROP TABLE IF EXISTS {tabela}")             # noqa: F821 (spark é global)
    for fn in (FUNC_STATUS_PEDIDO, FUNC_CALCULAR_FRETE):
        spark.sql(f"DROP FUNCTION IF EXISTS {fn}")              # noqa: F821
    print(f">> tabelas e funções removidas de {CATALOGO}.{SCHEMA}")
except NameError:
    print("(tabelas/funções) rode dentro de um workspace Databricks (precisa da sessão `spark`).")
except Exception as e:
    print(f"(tabelas/funções) sem permissão ou já removidas: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## (Opcional) Apagar também o catálogo
# MAGIC O `CASCADE` leva junto **tudo** que existir no catálogo — inclusive as tabelas de traces e o
# MAGIC que não for da aula. Descomente por sua conta e risco.

# COMMAND ----------

# spark.sql(f"DROP CATALOG IF EXISTS {CATALOGO} CASCADE")  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC Ambiente zerado — sobram apenas o experimento/traces do MLflow (histórico). Para reconstruir a
# MAGIC estrutura, rode o `00_setup_recursos`; para servir de novo, o Passo 5.
