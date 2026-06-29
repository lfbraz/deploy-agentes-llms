# Databricks notebook source
# MAGIC %md
# MAGIC # Passo 6 — O pipeline LLMOps de ponta a ponta
# MAGIC
# MAGIC Este passo amarra a aula inteira. LLMOps não é uma ferramenta, é um **ciclo de vida** que leva
# MAGIC o agente da bancada até produção com governança. Encadeamos os estágios — o que o Databricks
# MAGIC (MLflow + Unity Catalog) orquestra:
# MAGIC
# MAGIC ```
# MAGIC DESENVOLVER → REGISTRAR → GATE → PROMOVER → SERVIR → MONITORAR
# MAGIC ```
# MAGIC
# MAGIC Cada transição é registrada numa **trilha de auditoria**. É essa rastreabilidade — não só o
# MAGIC deploy — que caracteriza a governança corporativa.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ambiente
# MAGIC O mlflow do runtime do Databricks pode ser antigo demais para o `ResponsesAgent` e o
# MAGIC `mlflow.genai`. Instalamos as versões atuais no cluster e reiniciamos o Python.

# COMMAND ----------

# MAGIC %pip install -U mlflow databricks-langchain

# COMMAND ----------

# Reinicia o Python para carregar as versões recém-instaladas (apenas no Databricks).
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

from agente.agente_suporte import responder
from agente.config import CONFIG_PADRAO
from agente.mlflow_databricks import configurar_mlflow, credenciais_ok, nome_uc

if not credenciais_ok():
    raise SystemExit("Configure DATABRICKS_HOST/TOKEN no aula-07/.env (veja o .env.example).")

configurar_mlflow()
NOME_PROMPT = nome_uc("agente_suporte_prompt")
auditoria = []
def audit(etapa, detalhe):
    auditoria.append(f"[{etapa:10}] {detalhe}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Desenvolver e registrar
# MAGIC
# MAGIC O agente já funciona (Passo 1). Registramos a versão do prompt no Prompt Registry (Passo 2).

# COMMAND ----------

demo = responder("Qual a garantia dos eletrônicos?")
audit("DEV", f"agente respondeu via {', '.join(demo.ferramentas_usadas) or 'LLM'} "
             f"(trace {demo.trace_id})")

versao = mlflow.genai.register_prompt(
    name=NOME_PROMPT, template=CONFIG_PADRAO.system_prompt, commit_message="pipeline")
audit("REGISTRO", f"prompt v{versao.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate de aprovação e promoção
# MAGIC
# MAGIC Validamos se as respostas respeitam as instruções obrigatórias (Passo 3). Só promovemos o
# MAGIC alias `champion` se o conteúdo for válido.

# COMMAND ----------

INSTRUCOES = [
    ("Posso trocar um produto comprado há 20 dias?", "30 dias"),
    ("Qual a garantia dos eletrônicos?", "12 meses"),
    ("Qual o status do pedido 1002?", "1002"),
]
conformes = sum(termo.lower() in responder(p, CONFIG_PADRAO).texto.lower()
                for p, termo in INSTRUCOES)
conformidade = conformes / len(INSTRUCOES)
valido = conformidade >= 0.75
audit("CI/CD", f"v{versao.version}: conformidade={conformidade:.0%} -> "
               f"{'VÁLIDO' if valido else 'INVÁLIDO'}")

if valido:
    mlflow.genai.set_prompt_alias(NOME_PROMPT, alias="champion", version=versao.version)
    audit("PROMOÇÃO", f"champion -> v{versao.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Servir e monitorar

# COMMAND ----------

champ = mlflow.genai.load_prompt(f"prompts:/{NOME_PROMPT}@champion")
audit("SERVING", f"tráfego roteado para champion = v{champ.version}")

for q in ["Qual o status do pedido 1002?", "Posso trocar em 20 dias?"]:
    responder(q)
audit("MONITOR", "2 requisições rastreadas (traces/feedback)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trilha de auditoria (governança ponta a ponta)

# COMMAND ----------

print(*auditoria, sep="\n")
print(f"\n>> Produção está servindo: champion = v{champ.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC LLMOps é o ciclo — desenvolver, registrar, validar (gate), promover por alias, servir e
# MAGIC monitorar — com cada passo **auditado**. As peças das aulas anteriores (ferramentas, RAG,
# MAGIC guardrails, avaliação, observabilidade) viram estágios governados dessa única esteira.
