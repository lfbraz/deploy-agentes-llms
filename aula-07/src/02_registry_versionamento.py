# Databricks notebook source
# MAGIC %md
# MAGIC # Passo 2 — Registrar e versionar o agente (MLflow Prompt Registry)
# MAGIC
# MAGIC Com o agente pronto, o segundo passo é **versionar**. Diferente do MLOps clássico (onde se
# MAGIC versiona pesos de um modelo treinado), num agente LLM o artefato que mais muda é o **prompt**.
# MAGIC Usamos o **MLflow Prompt Registry** no **Unity Catalog**, que dá versionamento imutável estilo
# MAGIC Git + **aliases móveis**:
# MAGIC
# MAGIC - `register_prompt(name, template, ...)` cria uma versão **imutável** (v1, v2, v3...);
# MAGIC - `set_prompt_alias(name, alias, version)` move um alias (`champion`, `challenger`);
# MAGIC - `load_prompt("prompts:/<name>@<alias>")` é como o código de produção resolve a versão.
# MAGIC
# MAGIC Promover = mover o alias `champion`; rollback = mover o alias de volta. A versão é imutável; o
# MAGIC alias é o ponteiro. Essa indireção torna deploy e rollback uma operação O(1).

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

from agente.config import CONFIG_PADRAO
from agente.mlflow_databricks import configurar_mlflow, credenciais_ok, nome_uc

if not credenciais_ok():
    raise SystemExit("Configure DATABRICKS_HOST/TOKEN no aula-07/.env (veja o .env.example).")

configurar_mlflow()
NOME_PROMPT = nome_uc("agente_suporte_prompt")     # ex.: main.default.agente_suporte_prompt
print("prompt registrado em:", NOME_PROMPT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Três versões do prompt (a tripla evoluindo)
# MAGIC
# MAGIC Cada `register_prompt` no mesmo nome incrementa a versão. Guardamos a configuração de geração
# MAGIC (endpoint, temperatura) nas *tags* da versão.

# COMMAND ----------

def registrar_versao(template, autor):
    return mlflow.genai.register_prompt(
        name=NOME_PROMPT, template=template, commit_message=f"versão por {autor}",
        tags={"autor": autor, "endpoint": CONFIG_PADRAO.endpoint,
              "temperatura": str(CONFIG_PADRAO.temperatura)})

v1 = registrar_versao("Você é um atendente. Responda em 1 frase.", "luiz")
v2 = registrar_versao("Você é um atendente cordial. Responda em 1 frase e cite a política.", "luiz")
v3 = registrar_versao("Você é um atendente cordial. Responda em 1 frase, cite a política e "
                      "ofereça o próximo passo.", "ana")
print(f"versões registradas: v{v1.version}, v{v2.version}, v{v3.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aliases: o que está em produção
# MAGIC
# MAGIC O código de produção resolve pelo **alias**, nunca pelo número da versão.

# COMMAND ----------

def resolver(alias):
    return mlflow.genai.load_prompt(f"prompts:/{NOME_PROMPT}@{alias}")

mlflow.genai.set_prompt_alias(NOME_PROMPT, alias="champion", version=v2.version)
mlflow.genai.set_prompt_alias(NOME_PROMPT, alias="challenger", version=v3.version)
print("champion   -> v", resolver("champion").version)
print("challenger -> v", resolver("challenger").version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promover e fazer rollback (reapontar o ponteiro)

# COMMAND ----------

# O challenger venceu os testes -> promover a champion.
mlflow.genai.set_prompt_alias(NOME_PROMPT, alias="champion", version=v3.version)
print("após promover, champion -> v", resolver("champion").version)

# Incidente em produção -> rollback imediato para a versão anterior.
mlflow.genai.set_prompt_alias(NOME_PROMPT, alias="champion", version=v2.version)
print("após rollback, champion -> v", resolver("champion").version)

# COMMAND ----------

# MAGIC %md
# MAGIC Versione o prompt como artefato **imutável** e deixe aliases móveis decidirem o que vai a
# MAGIC produção. Deploy e rollback viram um reapontamento de ponteiro — a base de *blue-green* e
# MAGIC *champion/challenger*.
