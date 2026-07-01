# Databricks notebook source
# MAGIC %md
# MAGIC # Passo 2 — Versionar o prompt (MLflow Prompt Registry no Unity Catalog)
# MAGIC
# MAGIC Diferente do MLOps clássico (onde se versiona pesos de um modelo treinado), num agente LLM o
# MAGIC artefato que mais muda é o **prompt**. Usamos o **MLflow Prompt Registry** no **Unity Catalog**,
# MAGIC que dá versionamento imutável estilo Git + **aliases móveis**:
# MAGIC
# MAGIC - `register_prompt(name, template, ...)` cria uma versão **imutável** (v1, v2, v3...);
# MAGIC - o template usa a sintaxe **`{{variável}}`** — preenchida em tempo de execução com `format()`;
# MAGIC - `set_prompt_alias(name, alias, version)` move um alias (`champion`, `challenger`);
# MAGIC - `load_prompt("prompts:/<name>@<alias>")` é como o código de produção resolve a versão.
# MAGIC
# MAGIC Promover = mover o alias `champion`; rollback = mover o alias de volta. A versão é imutável; o
# MAGIC alias é o ponteiro. Essa indireção torna deploy e rollback uma operação O(1).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ambiente

# COMMAND ----------

# MAGIC %pip install -U mlflow databricks-langchain

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

import mlflow

from agente.config import LOJA, PROMPT_UC, SYSTEM_PROMPT_PADRAO
from agente.mlflow_databricks import configurar_mlflow, credenciais_ok

if not credenciais_ok():
    raise SystemExit("Configure DATABRICKS_HOST/TOKEN no aula-07/.env (veja o .env.example).")

configurar_mlflow()
print("prompt registrado em:", PROMPT_UC)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Três versões do prompt
# MAGIC
# MAGIC Cada `register_prompt` no mesmo nome incrementa a versão. O template mantém a variável
# MAGIC `{{loja}}`. Guardamos metadados úteis (autor, endpoint testado) nas *tags* da versão.

# COMMAND ----------

def registrar_versao(template, autor):
    return mlflow.genai.register_prompt(
        name=PROMPT_UC, template=template, commit_message=f"versão por {autor}",
        tags={"autor": autor, "tested_with": "databricks-claude-sonnet-4-5"})

v1 = registrar_versao("Você é o atendente da loja {{loja}}. Responda em 1 frase.", "luiz")
v2 = registrar_versao("Você é o atendente cordial da loja {{loja}}. Responda em 1 frase e cite "
                      "a política quando houver.", "luiz")
v3 = registrar_versao(SYSTEM_PROMPT_PADRAO, "ana")   # o prompt completo do agente (config.py)
print(f"versões registradas: v{v1.version}, v{v2.version}, v{v3.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aliases: o que está em produção
# MAGIC
# MAGIC O código de produção resolve pelo **alias**, nunca pelo número da versão. Ao carregar, usamos
# MAGIC `format(loja=...)` para preencher a variável `{{loja}}`.

# COMMAND ----------

mlflow.genai.set_prompt_alias(PROMPT_UC, alias="champion", version=v2.version)
mlflow.genai.set_prompt_alias(PROMPT_UC, alias="challenger", version=v3.version)

champion = mlflow.genai.load_prompt(f"prompts:/{PROMPT_UC}@champion")
challenger = mlflow.genai.load_prompt(f"prompts:/{PROMPT_UC}@challenger")
print("champion   -> v", champion.version)
print("challenger -> v", challenger.version)
print("\nprompt champion formatado:\n  ", champion.format(loja=LOJA)[:120], "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promover e fazer rollback (reapontar o ponteiro)

# COMMAND ----------

# O challenger se saiu melhor nos testes -> vamos promovê-lo a champion.
mlflow.genai.set_prompt_alias(PROMPT_UC, alias="champion", version=v3.version)
print("após promover, champion -> v",
      mlflow.genai.load_prompt(f"prompts:/{PROMPT_UC}@champion").version)

# Problema em produção -> rollback imediato: o alias volta para a versão anterior.
mlflow.genai.set_prompt_alias(PROMPT_UC, alias="champion", version=v2.version)
print("após rollback, champion -> v",
      mlflow.genai.load_prompt(f"prompts:/{PROMPT_UC}@champion").version)

# COMMAND ----------

# MAGIC %md
# MAGIC Versione o prompt como artefato **imutável** e deixe aliases móveis decidirem o que vai a
# MAGIC produção. Deploy e rollback viram um reapontamento de ponteiro — a base de *blue-green* e
# MAGIC *champion/challenger*. No serving (Passo 5), o agente carrega o prompt pelo alias `champion`.
