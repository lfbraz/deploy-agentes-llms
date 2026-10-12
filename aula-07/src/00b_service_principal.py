# Databricks notebook source
# MAGIC %md
# MAGIC # Passo 0b — A identidade do agente em produção (Service Principal)
# MAGIC
# MAGIC O endpoint servido (Passo 5) lê o system prompt do **Prompt Registry** a cada requisição, pelo
# MAGIC alias `champion`. Mas o prompt não entra na lista de `resources` do `log_model` — e são esses
# MAGIC `resources` que definem as credenciais automáticas do serving. Por isso o endpoint passa a rodar
# MAGIC com uma **identidade própria**: um **Service Principal** (SP), com o mínimo de permissões:
# MAGIC
# MAGIC - `USE CATALOG` no catálogo e `USE SCHEMA` + `EXECUTE` + `SELECT` no schema — cobre as UC
# MAGIC   Functions, a tabela de pedidos e o índice de Vector Search;
# MAGIC - `CREATE FUNCTION` + `MANAGE` no schema: o Prompt Registry exige esse trio (com o `EXECUTE`)
# MAGIC   até para **ler** um prompt;
# MAGIC - `CAN_USE` no endpoint de Vector Search e `CAN_EDIT` no experimento de produção (traces).
# MAGIC
# MAGIC As credenciais OAuth (M2M) do SP vão para um **secret scope**: o deploy só **referencia** os
# MAGIC segredos (`{{secrets/<scope>/<chave>}}`) — nenhum valor aparece no código. Idempotente.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ambiente

# COMMAND ----------

# MAGIC %pip install -U mlflow databricks-sdk

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

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, iam, ml

from agente.config import CATALOGO, ENDPOINT_VECTOR_SEARCH, SCHEMA, SECRET_SCOPE, SP_NOME
from agente.mlflow_databricks import credenciais_ok, usar_experimento_producao

if not credenciais_ok():
    raise SystemExit("Configure DATABRICKS_HOST/TOKEN no aula-07/.env (veja o .env.example).")

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Criar o Service Principal (ou reaproveitar o existente)

# COMMAND ----------

existentes = list(w.service_principals.list(filter=f'displayName eq "{SP_NOME}"'))
sp = existentes[0] if existentes else w.service_principals.create(
    display_name=SP_NOME, entitlements=[iam.ComplexValue(value="workspace-access")])
print(f">> service principal: {sp.display_name} (application_id {sp.application_id})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Guardar as credenciais OAuth num secret scope
# MAGIC O `client_secret` só é exibido na criação — por isso ele vai direto para o secret scope, sem
# MAGIC passar por nenhum `print`. Se o scope já tem as credenciais, não geramos outro segredo.

# COMMAND ----------

if SECRET_SCOPE not in {s.name for s in w.secrets.list_scopes()}:
    w.secrets.create_scope(scope=SECRET_SCOPE)

if "client_secret" not in {s.key for s in w.secrets.list_secrets(scope=SECRET_SCOPE)}:
    segredo = w.service_principal_secrets_proxy.create(service_principal_id=sp.id)
    w.secrets.put_secret(scope=SECRET_SCOPE, key="client_secret", string_value=segredo.secret)
w.secrets.put_secret(scope=SECRET_SCOPE, key="client_id", string_value=sp.application_id)
print(f">> segredos em '{SECRET_SCOPE}':", sorted(s.key for s in w.secrets.list_secrets(scope=SECRET_SCOPE)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Conceder o mínimo de permissões
# MAGIC Unity Catalog (prompt, funções, tabela e índice) + endpoint de Vector Search + experimento de
# MAGIC produção. Os endpoints de Foundation Model (`databricks-*`) já são acessíveis a todo o workspace.

# COMMAND ----------

def conceder(tipo, nome, *privilegios):
    w.grants.update(securable_type=tipo, full_name=nome, changes=[
        catalog.PermissionsChange(principal=sp.application_id, add=list(privilegios))])
    print(f">> {tipo} {nome}: {', '.join(p.value for p in privilegios)}")

P = catalog.Privilege
conceder("catalog", CATALOGO, P.USE_CATALOG)
# CREATE FUNCTION + MANAGE: exigidos pelo Prompt Registry até para LER o prompt. Num projeto real,
# isole os prompts num schema próprio para não estender esses privilégios às tabelas.
conceder("schema", f"{CATALOGO}.{SCHEMA}", P.USE_SCHEMA, P.EXECUTE, P.SELECT,
         P.CREATE_FUNCTION, P.MANAGE)

vs = w.vector_search_endpoints.get_endpoint(ENDPOINT_VECTOR_SEARCH)
w.permissions.update("vector-search-endpoints", vs.id, access_control_list=[
    iam.AccessControlRequest(service_principal_name=sp.application_id,
                             permission_level=iam.PermissionLevel.CAN_USE)])
print(f">> vector search {ENDPOINT_VECTOR_SEARCH}: CAN_USE")

exp = usar_experimento_producao()
w.experiments.update_permissions(exp.experiment_id, access_control_list=[
    ml.ExperimentAccessControlRequest(service_principal_name=sp.application_id,
                                      permission_level=ml.ExperimentPermissionLevel.CAN_EDIT)])
print(f">> experimento {exp.name}: CAN_EDIT")

# COMMAND ----------

# MAGIC %md
# MAGIC Pronto: o endpoint do Passo 5 vai rodar como este SP (`DATABRICKS_CLIENT_ID`/`SECRET` vindos do
# MAGIC secret scope) e ler `PROMPT_UC@champion` direto do Prompt Registry. Promover ou fazer rollback
# MAGIC do prompt passa a valer em produção **sem novo deploy**.
