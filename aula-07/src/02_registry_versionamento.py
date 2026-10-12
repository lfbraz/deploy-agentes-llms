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
from agente.prompt_registry import mover_alias, registrar_versao

if not credenciais_ok():
    raise SystemExit("Configure DATABRICKS_HOST/TOKEN no aula-07/.env (veja o .env.example).")

configurar_mlflow()
print("prompt registrado em:", PROMPT_UC)

# COMMAND ----------

# MAGIC %md
# MAGIC ## A versão em produção
# MAGIC O Passo 0 registrou a **primeira versão produtiva** e apontou o alias `champion` para ela. É por
# MAGIC esse alias que o agente carrega o prompt — o código nunca cita um número de versão.

# COMMAND ----------

producao = mlflow.genai.load_prompt(f"prompts:/{PROMPT_UC}@champion")
print(f"champion -> v{producao.version} | tags: {producao.tags}")
print("\nformatado:", producao.format(loja=LOJA)[:120], "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Uma nova versão candidata (challenger)
# MAGIC Alguém do time propõe um ajuste. `register_prompt` cria uma versão **imutável** nova (se o texto
# MAGIC for idêntico a uma versão existente, reaproveitamos a mesma). Ela entra como `challenger` — a
# MAGIC produção não muda. Comparamos frase a frase o que mudou entre as duas versões.

# COMMAND ----------

import difflib
import re

variante = SYSTEM_PROMPT_PADRAO + " Ao final, pergunte se o cliente precisa de mais alguma ajuda."
candidata = registrar_versao(variante, autor="Ana", mensagem="pergunta se pode ajudar em algo mais")
mover_alias("challenger", candidata)

def frases(texto):
    return re.split(r"(?<=[.!?])\s+", texto.strip())   # uma frase por linha, para o diff

for linha in difflib.unified_diff(frases(producao.template), frases(variante), lineterm="", n=0,
                                  fromfile=f"v{producao.version} (champion)",
                                  tofile=f"v{candidata} (challenger)"):
    print(linha)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promover e fazer rollback (reapontar o ponteiro)
# MAGIC Promover = mover o `champion` para a candidata; rollback = movê-lo de volta. O agente — local
# MAGIC ou no endpoint servido — resolve o alias a cada requisição, então a troca vale **sem novo
# MAGIC deploy** (o alias fica em cache por até 60 s).

# COMMAND ----------

def versao_champion():
    # cache_ttl_seconds=0: lê o alias na hora (o padrão guarda o alias em cache por 60 s)
    return mlflow.genai.load_prompt(f"prompts:/{PROMPT_UC}@champion", cache_ttl_seconds=0).version

mover_alias("champion", candidata)
print("após promover, champion -> v", versao_champion())

mover_alias("champion", producao.version)   # rollback: a produção volta para a versão anterior
print("após rollback, champion -> v", versao_champion())

# COMMAND ----------

# MAGIC %md
# MAGIC Versione o prompt como artefato **imutável** e deixe aliases móveis decidirem o que vai a
# MAGIC produção. Aqui a promoção foi manual, só para demonstrar; no fluxo real quem move o `champion`
# MAGIC é o pipeline (Passo 6), e só depois que o **gate** aprova a versão candidata.
