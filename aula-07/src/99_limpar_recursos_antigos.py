# Databricks notebook source
# MAGIC %md
# MAGIC # (Opcional) Limpeza — remover os recursos antigos de outro catálogo
# MAGIC
# MAGIC A Aula 07 passou a viver em `aula_07.default`. Se você rodou os exemplos antes, sobraram
# MAGIC recursos num catálogo antigo (por padrão `main.default`). Este notebook **apaga** esses
# MAGIC recursos para não deixar lixo pelo workspace:
# MAGIC
# MAGIC - as **UC Functions** (`consultar_status_pedido`, `calcular_frete`);
# MAGIC - a **tabela** de políticas e o **índice de Vector Search**;
# MAGIC - o **modelo registrado** e o **prompt** do Prompt Registry.
# MAGIC
# MAGIC > ⚠️ **É destrutivo.** Ele NÃO faz parte do provisionamento (`00_setup`) — execute-o
# MAGIC > manualmente, uma única vez. Por segurança, ele **se recusa a apagar o catálogo novo** (o que está no `.env`).
# MAGIC > Ajuste o catálogo/schema antigos em `UC_CATALOG_ANTIGO`/`UC_SCHEMA_ANTIGO` se preciso.

# COMMAND ----------

# MAGIC %pip install -U mlflow databricks-langchain databricks-vectorsearch unitycatalog-ai

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

from agente.config import CATALOGO, ENDPOINT_VECTOR_SEARCH, SCHEMA
from agente.mlflow_databricks import configurar_mlflow, credenciais_ok

if not credenciais_ok():
    raise SystemExit("Configure DATABRICKS_HOST/TOKEN no aula-07/.env (veja o .env.example).")

configurar_mlflow()

# Catálogo/schema ANTIGOS a limpar (o padrão anterior era main.default).
CAT_ANTIGO = os.getenv("UC_CATALOG_ANTIGO", "main")
SCHEMA_ANTIGO = os.getenv("UC_SCHEMA_ANTIGO", "default")
ANTIGO = f"{CAT_ANTIGO}.{SCHEMA_ANTIGO}"

# Trava de segurança: nunca apagar o catálogo/schema ATUAL da aula.
if (CAT_ANTIGO, SCHEMA_ANTIGO) == (CATALOGO, SCHEMA):
    raise SystemExit(f"Recusado: {ANTIGO} é o catálogo ATUAL da aula. Aponte "
                     f"UC_CATALOG_ANTIGO/UC_SCHEMA_ANTIGO para o catálogo que você quer limpar.")

# Os objetos que a aula cria (mesmos nomes, no catálogo antigo).
FUNCS = [f"{ANTIGO}.consultar_status_pedido", f"{ANTIGO}.calcular_frete"]
TABELA = f"{ANTIGO}.politicas"
INDICE = f"{ANTIGO}.politicas_idx"
MODELO = f"{ANTIGO}.agente_suporte"
PROMPT = f"{ANTIGO}.agente_suporte_prompt"

print(f"vou remover os recursos da aula em: {ANTIGO}")
for x in [*FUNCS, TABELA, INDICE, MODELO, PROMPT]:
    print("   -", x)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Índice de Vector Search
# MAGIC O índice não é apagado por SQL — usamos o `VectorSearchClient`.

# COMMAND ----------

try:
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient(disable_notice=True)
    try:
        vsc.delete_index(index_name=INDICE)
    except TypeError:
        vsc.delete_index(endpoint_name=ENDPOINT_VECTOR_SEARCH, index_name=INDICE)
    print(f">> índice removido: {INDICE}")
except Exception as e:
    print(f"(índice) nada a remover ou sem permissão: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tabela e UC Functions
# MAGIC Aqui um `DROP ... IF EXISTS` resolve (via `spark`).

# COMMAND ----------

try:
    spark.sql(f"DROP TABLE IF EXISTS {TABELA}")                 # noqa: F821 (spark é global)
    for fn in FUNCS:
        spark.sql(f"DROP FUNCTION IF EXISTS {fn}")              # noqa: F821
    print(f">> tabela e funções removidas de {ANTIGO}")
except NameError:
    print("(tabela/funções) rode dentro de um workspace Databricks (precisa da sessão `spark`).")
except Exception as e:
    print(f"(tabela/funções) sem permissão ou já removidas: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelo registrado e prompt
# MAGIC Ambos são objetos do Unity Catalog — apagamos pelo `MlflowClient`. Para o prompt, é preciso
# MAGIC apagar as versões antes do objeto.

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

try:
    client.delete_registered_model(MODELO)
    print(f">> modelo removido: {MODELO}")
except Exception as e:
    print(f"(modelo) nada a remover ou sem permissão: {e}")

try:
    # Apaga as versões (do UC exige remover as versões antes do prompt) e depois o prompt.
    for v in range(1, 50):
        try:
            client.delete_prompt_version(PROMPT, str(v))
        except Exception:
            pass
    client.delete_prompt(PROMPT)
    print(f">> prompt removido: {PROMPT}")
except Exception as e:
    print(f"(prompt) nada a remover, versões restantes ou sem permissão: {e}\n"
          f"        se necessário, apague pelo Catalog Explorer em {ANTIGO}.")

# COMMAND ----------

# MAGIC %md
# MAGIC Pronto — o catálogo antigo (`main.default`) fica sem os recursos da Aula 07. Tudo o que a aula
# MAGIC usa passa a viver só em `aula_07.default` (rode o `00_setup_recursos` para provisioná-los lá).
