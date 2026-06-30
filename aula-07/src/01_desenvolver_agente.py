# Databricks notebook source
# MAGIC %md
# MAGIC # Passo 1 — Desenvolver o agente: tools (UC Functions), RAG e observabilidade
# MAGIC
# MAGIC A jornada de LLMOps começa pelo agente. Reunimos três pilares das aulas anteriores sobre os
# MAGIC primitivos **nativos do Databricks**:
# MAGIC
# MAGIC - **tool use** — o agente decide chamar **UC Functions** (status do pedido, frete) — Aulas 3-4;
# MAGIC - **RAG** — recupera a política da loja num índice de **Vector Search** antes de responder — Aula 4;
# MAGIC - **observabilidade** — `mlflow.langchain.autolog()` transforma cada chamada ao modelo e a cada
# MAGIC   ferramenta em *spans* de uma **trace** no MLflow — Aula 6.
# MAGIC
# MAGIC O agente é autorado na interface **`ResponsesAgent` do MLflow** (Databricks Agent Framework),
# MAGIC com um Claude servido pelo Databricks como modelo. **Pré-requisito:** rode antes o Passo 0
# MAGIC (`00_setup_recursos.py`), que cria as UC Functions e o índice.

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

from agente.agente_suporte import TELEMETRIA, responder
from agente.base_vetorial import backend_rag
from agente.config import CONFIG_PADRAO
from agente.mlflow_databricks import credenciais_ok

if not credenciais_ok():
    raise SystemExit("Configure DATABRICKS_HOST/TOKEN no aula-07/.env (veja o .env.example).")

print("endpoint do modelo:", CONFIG_PADRAO.endpoint)
print("índice (RAG)      :", backend_rag())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Uma conversa cobrindo cada pilar
# MAGIC
# MAGIC O `responder()` roda o agente e devolve a telemetria da requisição (texto, ferramentas usadas,
# MAGIC latência e o `trace_id`). Observe a coluna *ferramentas*: a UC Function entra nas perguntas de
# MAGIC pedido/frete e o `buscar_politica` (Vector Search) entra nas perguntas de política.

# COMMAND ----------

roteiro = [
    ("UC Function: status do pedido", "Qual o status do pedido 1002?"),
    ("UC Function: frete", "Quanto fica o frete para o CEP 01000-000, 2 kg?"),
    ("RAG (Vector Search): troca", "Posso trocar um produto comprado há 20 dias?"),
    ("RAG (Vector Search): garantia", "Qual a garantia dos eletrônicos?"),
]
for rotulo, pergunta in roteiro:
    r = responder(pergunta)
    print(f"\n[{rotulo}]")
    print(f"  Usuário : {pergunta}")
    print(f"  Agente  : {r.texto}")
    tools = ", ".join(r.ferramentas_usadas) or "—"
    print(f"  Trace   : {r.trace_id} | {r.latencia_ms:.0f} ms | ferramentas: {tools}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Observabilidade pelo autolog
# MAGIC
# MAGIC Como o `autolog` está ligado, cada requisição acima virou uma **trace** com os *spans* do LLM e
# MAGIC das ferramentas — visível na aba **Traces** do experimento/modelo no MLflow. Aqui sintetizamos a
# MAGIC sessão; o detalhe (entradas, saídas, latência por span) fica na UI.

# COMMAND ----------

total = len(TELEMETRIA)
lat_media = sum(t.latencia_ms for t in TELEMETRIA) / total
tools = [f for t in TELEMETRIA for f in t.ferramentas_usadas]
print(f"requisições     : {total}")
print(f"latência média  : {lat_media:.0f} ms")
print(f"chamadas de tool: {len(tools)} ({', '.join(sorted(set(tools))) or '—'})")
print("\n>> Abra a aba 'Traces' no MLflow para inspecionar os spans de cada requisição.")

# COMMAND ----------

# MAGIC %md
# MAGIC O agente é uma **solução de software completa** — UC Functions, RAG por Vector Search e
# MAGIC observabilidade automática juntos. É ele que os próximos passos vão versionar (prompt registry),
# MAGIC avaliar com feedback humano, barrar num gate de aprovação, servir e operar.
