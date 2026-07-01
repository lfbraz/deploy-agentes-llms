# Databricks notebook source
# MAGIC %md
# MAGIC # Passo 0 — Provisionar os recursos governados no Unity Catalog
# MAGIC
# MAGIC Antes de desenvolver o agente, criamos os **recursos que ele consome** — todos governados pelo
# MAGIC Unity Catalog, do jeito que o **Databricks Agent Framework** recomenda:
# MAGIC
# MAGIC - **Tabela de pedidos**: uma tabela Delta com os pedidos da loja — os **dados reais** que a
# MAGIC   função de status consulta;
# MAGIC - **UC Functions** (as *tools*): `consultar_status_pedido` (função **SQL** que lê a tabela de
# MAGIC   pedidos) e `calcular_frete` (função **Python**), registradas no Unity Catalog (com
# MAGIC   permissões, linhagem e descoberta) — Aulas 3-4;
# MAGIC - **Índice de Vector Search** (o RAG): uma tabela Delta com as políticas da loja, sincronizada
# MAGIC   para um índice de **Databricks AI Search** com embeddings gerenciados — Aula 4.
# MAGIC
# MAGIC Este passo é **idempotente**: rode quantas vezes quiser. Ele precisa de um catálogo/schema com
# MAGIC permissão de escrita (configure `UC_CATALOG`/`UC_SCHEMA` no `.env`; padrão `aula_07.default`).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ambiente
# MAGIC Instalamos as versões atuais das bibliotecas e reiniciamos o Python (no Databricks).

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

from agente.config import (
    CATALOGO, ENDPOINT_EMBEDDINGS, ENDPOINT_VECTOR_SEARCH, INDICE_POLITICAS, SCHEMA,
    TABELA_PEDIDOS, TABELA_POLITICAS,
)
from agente.mlflow_databricks import credenciais_ok

if not credenciais_ok():
    raise SystemExit("Configure DATABRICKS_HOST/TOKEN no aula-07/.env (veja o .env.example).")

print(f"catálogo.schema : {CATALOGO}.{SCHEMA}")
print(f"tabela pedidos  : {TABELA_PEDIDOS}")
print(f"tabela políticas: {TABELA_POLITICAS}")
print(f"índice (RAG)    : {INDICE_POLITICAS}  (endpoint {ENDPOINT_VECTOR_SEARCH})")
print(f"embeddings      : {ENDPOINT_EMBEDDINGS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0) Criar o catálogo e o schema (se ainda não existirem)
# MAGIC
# MAGIC Tudo da Aula 07 mora em `aula_07.default`. Criamos o catálogo/schema aqui — idempotente, e no
# MAGIC Free Edition você costuma ter permissão para isso (é o dono do próprio workspace).

# COMMAND ----------

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOGO}")            # noqa: F821 (spark é global)
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOGO}.{SCHEMA}")    # noqa: F821
    print(f">> catálogo/schema prontos: {CATALOGO}.{SCHEMA}")
except NameError:
    raise SystemExit("Rode este passo dentro de um workspace Databricks (a sessão `spark` "
                     "precisa existir) ou via databricks-connect.")
except Exception as e:
    print(f"aviso: não consegui criar {CATALOGO}.{SCHEMA} (talvez sem permissão): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Criar a tabela de pedidos (os dados reais da função de status)
# MAGIC
# MAGIC A UC Function `consultar_status_pedido` é uma **função SQL** que consulta esta tabela — por
# MAGIC isso ela vem primeiro (o `CREATE FUNCTION` valida a referência à tabela). É também nela que a
# MAGIC demo do gate (`98_demo_gate_cicd.py`) mexe para fazer o CI/CD falhar ou passar.

# COMMAND ----------

from agente.ferramentas import PEDIDOS, criar_tabela_pedidos

try:
    tabela_pedidos = criar_tabela_pedidos(spark)  # noqa: F821 (spark é global no Databricks)
    print(f">> tabela {tabela_pedidos} criada com {len(PEDIDOS)} pedidos.")
except NameError:
    raise SystemExit("Este passo cria recursos reais — rode dentro de um workspace Databricks "
                     "(a sessão `spark` precisa existir) ou via databricks-connect.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Registrar as UC Functions (as ferramentas do agente)
# MAGIC
# MAGIC A consulta de status vira uma **função SQL** (`CREATE FUNCTION ... READS SQL DATA`) — funções
# MAGIC Python de UC rodam num sandbox sem acesso a tabelas. O cálculo de frete segue como **função
# MAGIC Python**: o `create_python_function` serializa a função (type hints + docstring) para o
# MAGIC Unity Catalog. Registradas, ambas são tools reutilizáveis.

# COMMAND ----------

from agente.ferramentas import registrar_uc_functions

funcs = registrar_uc_functions()
print(">> UC Functions registradas:")
for f in funcs:
    print("   -", f)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Criar a tabela de políticas (a fonte do RAG)
# MAGIC
# MAGIC O índice delta-sync se mantém em dia com uma tabela Delta. Gravamos as políticas e habilitamos
# MAGIC o **Change Data Feed** (pré-requisito do delta-sync). Precisa da sessão `spark` (Databricks).

# COMMAND ----------

from agente.base_vetorial import POLITICAS, criar_tabela_politicas

try:
    tabela = criar_tabela_politicas(spark)  # noqa: F821  (spark é global no Databricks)
    print(f">> tabela {tabela} criada com {len(POLITICAS)} políticas (CDF habilitado).")
except NameError:
    raise SystemExit("Este passo cria recursos reais — rode dentro de um workspace Databricks "
                     "(a sessão `spark` precisa existir) ou via databricks-connect.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Criar o endpoint e o índice de Vector Search
# MAGIC
# MAGIC O `create_delta_sync_index` cria um índice com **embeddings gerenciados**: declaramos a coluna
# MAGIC de texto e o endpoint de embeddings, e o Databricks calcula/mantém os vetores. A criação leva
# MAGIC alguns minutos na primeira vez (as próximas execuções apenas confirmam que já existe).

# COMMAND ----------

from agente.base_vetorial import criar_indice

indice = criar_indice()
print(f">> índice de Vector Search pronto: {indice}")

# COMMAND ----------

# MAGIC %md
# MAGIC Com as **UC Functions** e o **índice de Vector Search** no Unity Catalog, os recursos do agente
# MAGIC estão governados e prontos. O próximo passo monta o agente que os consome (tools + RAG) com
# MAGIC observabilidade automática.
