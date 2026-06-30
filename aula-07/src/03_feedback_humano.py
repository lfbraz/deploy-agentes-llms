# Databricks notebook source
# MAGIC %md
# MAGIC # Passo 3 — Feedback humano nas respostas do agente (MLflow 3)
# MAGIC
# MAGIC Em produção, quem melhor diz se a resposta foi boa é **o usuário**. O MLflow 3 captura esse
# MAGIC veredito como uma **assessment** do tipo *Feedback*, anexada à **trace** daquela requisição:
# MAGIC
# MAGIC - cada resposta tem um `trace_id` (do `autolog`);
# MAGIC - `mlflow.log_feedback(trace_id, name, value, source=AssessmentSource("HUMAN", ...))` grava o
# MAGIC   👍/👎 (e um comentário opcional) ligado àquela trace;
# MAGIC - o feedback alimenta a avaliação e a curadoria de *golden sets* (Aula 6) — e, adiante, o gate.
# MAGIC
# MAGIC É o recurso de **human feedback**: a voz do usuário vira sinal versionado e auditável, não um
# MAGIC e-mail perdido.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ambiente

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

from agente.agente_suporte import registrar_feedback, responder
from agente.mlflow_databricks import credenciais_ok

if not credenciais_ok():
    raise SystemExit("Configure DATABRICKS_HOST/TOKEN no aula-07/.env (veja o .env.example).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Responder e coletar o 👍/👎
# MAGIC
# MAGIC Rodamos algumas perguntas; para cada uma, registramos o feedback do usuário pela `trace_id`.
# MAGIC Aqui simulamos o veredito (na sua app real, ele vem do clique do usuário).

# COMMAND ----------

interacoes = [
    ("Qual a garantia dos eletrônicos?", True, "respondeu certo: 12 meses"),
    ("Qual o status do pedido 1002?", True, "trouxe o status do pedido"),
    ("Vocês parcelam em quantas vezes?", False, "não soube responder"),
]

for pergunta, util, comentario in interacoes:
    r = responder(pergunta)
    registrar_feedback(r, util=util, comentario=comentario, usuario="cliente@acme.com")
    print(f"{'👍' if util else '👎'} trace {r.trace_id} | {pergunta}")
    print(f"    agente: {r.texto}\n")

print(">> Feedbacks gravados como assessments HUMAN nas traces (aba 'Traces' no MLflow).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Como fica na sua aplicação (produção)
# MAGIC
# MAGIC A app devolve o `trace_id` junto da resposta e expõe um endpoint que recebe o clique do botão de
# MAGIC 👍/👎. O padrão recomendado pela documentação:
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC from mlflow.entities import AssessmentSource
# MAGIC from fastapi import FastAPI
# MAGIC
# MAGIC app = FastAPI()
# MAGIC
# MAGIC @app.post("/feedback")
# MAGIC def receber_feedback(trace_id: str, util: bool, comentario: str | None = None):
# MAGIC     mlflow.log_feedback(
# MAGIC         trace_id=trace_id, name="user_feedback", value=util,
# MAGIC         source=AssessmentSource(source_type="HUMAN", source_id="user@empresa.com"),
# MAGIC         rationale=comentario)
# MAGIC     return {"status": "ok"}
# MAGIC ```
# MAGIC
# MAGIC O 👎 marca traces para revisão e curadoria — as respostas ruins viram casos do **gate de
# MAGIC aprovação** (Passo 4), fechando o ciclo avaliação → CI/CD.
