# Databricks notebook source
# MAGIC %md
# MAGIC # Demo — o gate de aprovação decidindo o destino do pipeline (CI/CD)
# MAGIC
# MAGIC O gate (Passo 4) valida o agente real em situações conhecidas — e uma delas depende de
# MAGIC **dados de verdade**: o `test_status_pedido_usa_uc_function` pergunta pelo pedido **1002**, e a
# MAGIC UC Function `consultar_status_pedido` (função SQL) busca a resposta na tabela
# MAGIC `aula_07.default.pedidos` do Unity Catalog.
# MAGIC
# MAGIC Nesta demo, quebramos e consertamos essa premissa de propósito:
# MAGIC
# MAGIC 1. rodamos o gate com a tabela íntegra → **verde** (o merge seguiria);
# MAGIC 2. **removemos** o pedido 1002 da tabela e rodamos de novo → **vermelho** (*exit code* != 0 —
# MAGIC    no GitHub Actions, o merge seria bloqueado);
# MAGIC 3. **recolocamos** o registro → o gate volta a **passar**.
# MAGIC
# MAGIC É o mesmo `pytest` que a esteira executa (`ci/ci-cd-agente.yml`): o *exit code* do gate é quem
# MAGIC decide se o pipeline segue.
# MAGIC
# MAGIC **Pré-requisitos:** Passo 0 (recursos provisionados) e a sessão `spark` (workspace Databricks).

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

from agente.config import TABELA_PEDIDOS
from agente.mlflow_databricks import credenciais_ok

if not credenciais_ok():
    raise SystemExit("Configure DATABRICKS_HOST/TOKEN no aula-07/.env (veja o .env.example).")

import pytest


def rodar_gate() -> int:
    """Executa o mesmo pytest da esteira de CI/CD e devolve o exit code."""
    # No /Workspace o pytest não consegue gravar __pycache__/.pytest_cache (Errno 95): desligamos
    # essas escritas, senão a coleta falha antes de qualquer teste rodar.
    sys.dont_write_bytecode = True
    codigo = int(pytest.main([os.path.join(BASE, "src", "04_gate_aprovacao.py"), "-q",
                              "-p", "no:cacheprovider",
                              # plugins (langsmith/anyio) já importados pelo notebook: aviso inofensivo
                              "-W", "ignore::pytest.PytestAssertRewriteWarning"]))
    veredicto = "VERDE ✅ (merge liberado)" if codigo == 0 else "VERMELHO ❌ (merge bloqueado)"
    print(f"\n>> gate: exit={codigo} -> {veredicto}")
    return codigo


def mostrar_pedidos():
    spark.sql(f"SELECT * FROM {TABELA_PEDIDOS} ORDER BY numero").show(truncate=False)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Gate com a base íntegra → verde
# MAGIC
# MAGIC O pedido 1002 está na tabela, a UC Function devolve o status "a caminho" e todos os testes
# MAGIC passam — o pipeline seguiria para o deploy.

# COMMAND ----------

mostrar_pedidos()
assert rodar_gate() == 0, "esperava o gate verde com a tabela íntegra"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Removendo o pedido 1002 → o gate fica vermelho
# MAGIC
# MAGIC Simulamos uma regressão de **dados**: o registro que o teste conhece some da tabela. A UC
# MAGIC Function passa a responder "não encontrado", o `test_status_pedido_usa_uc_function` falha e o
# MAGIC `pytest` sai com *exit code* != 0 — exatamente o sinal que **bloqueia o merge** na esteira.

# COMMAND ----------

spark.sql(f"DELETE FROM {TABELA_PEDIDOS} WHERE numero = '1002'")  # noqa: F821
mostrar_pedidos()
assert rodar_gate() != 0, "esperava o gate vermelho sem o pedido 1002"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Recolocando o registro → o gate volta a passar
# MAGIC
# MAGIC Corrigida a base, o mesmo gate volta a ficar verde — nenhuma mudança de código foi necessária.

# COMMAND ----------

spark.sql(  # noqa: F821
    f"INSERT INTO {TABELA_PEDIDOS} VALUES "
    "('1002', 'a caminho — previsão de entrega em 2 dias úteis')"
)
mostrar_pedidos()
assert rodar_gate() == 0, "esperava o gate verde após restaurar o pedido 1002"

# COMMAND ----------

# MAGIC %md
# MAGIC O gate de aprovação é **binário e auditável**: verde libera, vermelho barra — seja a regressão
# MAGIC de código, de prompt ou de **dados**. No GitHub Actions (`ci/ci-cd-agente.yml`), este mesmo
# MAGIC `pytest` roda a cada *pull request*; reproduza a demo removendo o registro e disparando o
# MAGIC workflow para ver a esteira falhar de verdade.
