# Databricks notebook source
# MAGIC %md
# MAGIC # Passo 6 — O pipeline LLMOps de ponta a ponta
# MAGIC
# MAGIC Este passo amarra a aula. LLMOps não é uma ferramenta, é um **ciclo de vida** que leva o agente
# MAGIC da bancada até produção com governança. Encadeamos os estágios que o Databricks (MLflow + Unity
# MAGIC Catalog) orquestra:
# MAGIC
# MAGIC ```
# MAGIC SETUP → REGISTRAR PROMPT → GATE (testes) → PROMOVER ALIAS → SERVIR → FEEDBACK/MONITORAR
# MAGIC ```
# MAGIC
# MAGIC Cada transição é registrada numa **trilha de auditoria**. É essa rastreabilidade — não só o
# MAGIC deploy — que caracteriza a governança corporativa. A versão automatizada disto é a esteira de
# MAGIC CI/CD (`ci/ci-cd-agente.yml`), onde o **gate** decide se o merge segue.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ambiente

# COMMAND ----------

# MAGIC %pip install -U mlflow databricks-langchain databricks-vectorsearch unitycatalog-ai pytest

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

from agente.agente_suporte import registrar_feedback, responder
from agente.config import MODELO_UC, PROMPT_UC, SYSTEM_PROMPT_PADRAO
from agente.mlflow_databricks import configurar_mlflow, credenciais_ok
from agente.prompt_registry import mover_alias, registrar_versao

if not credenciais_ok():
    raise SystemExit("Configure DATABRICKS_HOST/TOKEN no aula-07/.env (veja o .env.example).")

configurar_mlflow()
auditoria = []
def audit(etapa, detalhe):
    auditoria.append(f"[{etapa:10}] {detalhe}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registrar a versão candidata do prompt
# MAGIC
# MAGIC O agente já funciona (Passo 1) sobre os recursos do Passo 0. Registramos o prompt do código
# MAGIC (com a variável `{{loja}}`) como **candidato**: se o texto não mudou, reaproveitamos a versão
# MAGIC existente em vez de criar uma cópia. A candidata recebe o alias `challenger`.

# COMMAND ----------

candidata = registrar_versao(SYSTEM_PROMPT_PADRAO, mensagem="pipeline")
mover_alias("challenger", candidata)
audit("REGISTRO", f"prompt {PROMPT_UC} v{candidata} -> challenger")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate de aprovação (os testes unitários do Passo 4)
# MAGIC
# MAGIC Rodamos o **mesmo** gate da esteira: o `pytest` sobre `04_gate_aprovacao.py` — mas com
# MAGIC `PROMPT_ALIAS=challenger`, para o agente carregar a **versão candidata**. Assim o gate testa
# MAGIC exatamente o que vai ser promovido. Só promovemos se todos os testes passarem (exit code 0).

# COMMAND ----------

import pytest

# No /Workspace o pytest não consegue gravar __pycache__/.pytest_cache (Errno 95): desligamos
# essas escritas, senão a coleta falha antes de qualquer teste rodar.
sys.dont_write_bytecode = True
os.environ["PROMPT_ALIAS"] = "challenger"   # o agente do gate carrega a versão candidata
try:
    codigo = pytest.main([os.path.join(BASE, "src", "04_gate_aprovacao.py"), "-q",
                          "-p", "no:cacheprovider",
                          # plugins (langsmith/anyio) já importados pelo notebook: aviso inofensivo
                          "-W", "ignore::pytest.PytestAssertRewriteWarning"])
finally:
    os.environ.pop("PROMPT_ALIAS", None)    # de volta ao champion
valido = (codigo == 0)
audit("GATE", f"pytest exit={int(codigo)} -> {'VÁLIDO' if valido else 'INVÁLIDO'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promover (mover o alias champion)
# MAGIC
# MAGIC Gate verde -> o alias `champion` aponta para esta versão. Gate vermelho -> nada é promovido (e,
# MAGIC na esteira, o merge é bloqueado).

# COMMAND ----------

if valido:
    mover_alias("champion", candidata)
    audit("PROMOÇÃO", f"champion -> v{candidata}")
else:
    audit("PROMOÇÃO", "bloqueada (gate vermelho)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Servir e coletar feedback
# MAGIC
# MAGIC O deploy real está no Passo 5 (`agents.deploy(MODELO_UC, versão)`). Aqui registramos a intenção
# MAGIC na trilha e exercitamos o agente coletando feedback (Passo 3) — o sinal que realimenta o gate.

# COMMAND ----------

if valido:
    audit("SERVING", f"deploy de {MODELO_UC} (ver Passo 5: agents.deploy)")
    r = responder("Qual o status do pedido 1002?")
    registrar_feedback(r, util=True, comentario="status correto")
    audit("MONITOR", f"1 requisição rastreada + feedback (trace {r.trace_id})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trilha de auditoria (governança ponta a ponta)

# COMMAND ----------

print(*auditoria, sep="\n")
print(f"\n>> Gate: {'VERDE ✅ — pipeline seguiu até produção' if valido else 'VERMELHO ❌ — barrado'}")

# COMMAND ----------

# MAGIC %md
# MAGIC LLMOps é o ciclo — setup, registrar o prompt, validar (gate), promover por alias, servir e
# MAGIC monitorar com feedback — cada passo **auditado**. As peças das aulas anteriores (tools, RAG,
# MAGIC observabilidade, avaliação) viram estágios governados de uma única esteira.
