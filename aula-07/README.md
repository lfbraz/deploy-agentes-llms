# Aula 07 — LLMOps: a jornada completa de um agente em produção

Nesta aula construímos um **agente de suporte de e-commerce** sobre os primitivos **nativos do
Databricks Agent Framework** e o levamos por toda a disciplina de **LLMOps** — o ciclo de vida
operacional que transforma um protótipo num produto:

```
SETUP → DESENVOLVER → VERSIONAR → FEEDBACK → GATE → SERVIR → ORQUESTRAR (CI/CD)
```

O que demonstramos, ponta a ponta:

1. **Agente com tools** — ferramentas como **UC Functions** (Unity Catalog) via `UCFunctionToolkit`;
   o status do pedido é uma **função SQL** que consulta dados reais na tabela `pedidos`;
2. **Observabilidade** — `mlflow.langchain.autolog()` (cada chamada vira *span* numa trace);
3. **RAG com base vetorial** — índice de **Databricks AI Search (Vector Search)** + `VectorSearchRetrieverTool`;
4. **Feedback humano** — `mlflow.log_feedback` anexa o 👍/👎 do usuário à trace;
5. **Gate de aprovação** — **testes unitários** (pytest) do comportamento do agente;
6. **Prompt registry** — versionamento do prompt no **MLflow Prompt Registry** (Unity Catalog) com aliases;
7. **Serving endpoint** — `log_model` (com `resources`) → registro no UC → `agents.deploy`;
8. **CI/CD** — **GitHub Actions** com o gate de aprovação barrando o merge.

> Os **guardrails** de produção e os **painéis de monitoramento** são configurados na própria UI
> do workspace (AI Gateway / Guardrails, *Inference Tables*, dashboards). No código, o
> comportamento é validado pelo **gate de aprovação** (testes unitários sobre situações conhecidas).

> Sem `DATABRICKS_HOST`/`DATABRICKS_TOKEN`, cada exemplo exibe os pré-requisitos e encerra. A Aula 07
> roda **sempre** contra o Databricks (não há modo offline).

---

## 1. O agente (`agente/`) — a peça central

O agente é uma **solução de software completa**, reunindo os pilares das aulas anteriores sobre
recursos governados:

| Arquivo | Pilar (aula) | O que faz |
|---------|--------------|-----------|
| [`config.py`](agente/config.py) | versionamento/infra | a "tripla" (prompt+endpoint+geração) + nomes UC + endpoints |
| [`prompt_registry.py`](agente/prompt_registry.py) | versionamento | o system prompt no **Prompt Registry**: versões imutáveis + aliases `champion`/`challenger` |
| [`ferramentas.py`](agente/ferramentas.py) | tool use (Aulas 3-4) | **UC Functions** (SQL sobre a tabela `pedidos` + Python) + `UCFunctionToolkit` |
| [`base_vetorial.py`](agente/base_vetorial.py) | RAG (Aula 4) | índice de **Vector Search** + `VectorSearchRetrieverTool` |
| [`agente_suporte.py`](agente/agente_suporte.py) | observabilidade (Aula 6) | `ResponsesAgent` (tools + RAG) + `autolog` + prompt pelo alias + feedback |
| [`mlflow_databricks.py`](agente/mlflow_databricks.py) | infra | pré-requisitos, tracking/registry no Unity Catalog |
| [`serving_model.py`](agente/serving_model.py) | deploy | entrypoint *models-from-code* p/ `log_model` / `agents.deploy` |

---

## 2. Pré-requisitos e configuração (obrigatório)

1. **Python 3.12** e as dependências (a partir da raiz do repositório):

   ```bash
   pip install -r requirements.txt   # mlflow, databricks-langchain, databricks-vectorsearch,
                                      # unitycatalog-ai, pytest (databricks-agents é instalado no notebook)
   ```

2. **Credenciais e recursos do Databricks** — copie o template e preencha:

   ```bash
   cp aula-07/.env.example aula-07/.env
   ```

   No `aula-07/.env`, defina:
   - `DATABRICKS_HOST` e `DATABRICKS_TOKEN` (workspace + Personal Access Token);
   - `UC_CATALOG`/`UC_SCHEMA` — catálogo.schema onde tudo mora (padrão `aula_07.default`). O
     `00_setup` cria o catálogo/schema se faltarem (`CREATE CATALOG IF NOT EXISTS`);
   - `ENDPOINT_LLM_AGENTE` — endpoint que suporte **tool calling** (ex.: `databricks-claude-sonnet-4-5`).
     **Não** use o `databricks-gpt-oss-20b` (não faz function calling);
   - `ENDPOINT_EMBEDDINGS` — endpoint de embeddings do índice de Vector Search;
   - `ENDPOINT_VECTOR_SEARCH` — nome do endpoint (compute) de Vector Search.

   Para os **traces/feedback** (exemplos 01/03) irem para tabelas no Unity Catalog (`aula_07.default`,
   prefixo `experimento_aula_07`), ligue o preview **"OpenTelemetry on Databricks"** e informe um
   SQL warehouse em `MLFLOW_TRACING_SQL_WAREHOUSE_ID`. Sem isso, os traces caem num experimento comum
   de workspace (`/Users/<você>/experimento_aula_07`) — o resto da aula funciona igual.

> O `.env` (com suas credenciais) **nunca** é versionado — ele está no `.gitignore`.

---

## 3. Passo a passo — rodando a jornada

Execute **na ordem** (a partir da raiz do repositório). Cada um é uma etapa do ciclo de vida.

### Passo 0 — Provisionar os recursos (UC Functions + Vector Search + prompt)

```bash
python aula-07/src/00_setup_recursos.py
```

Idempotente: cria a **tabela de pedidos** (os dados reais da função de status), registra as
**UC Functions** (tools), cria a tabela de políticas + o **índice de Vector Search** (RAG) e
registra a **primeira versão produtiva do prompt** no Prompt Registry, com o alias `champion`.
Roda **dentro de um workspace Databricks** (precisa da sessão `spark`).

### Passo 0b — A identidade do agente em produção (Service Principal)

```bash
python aula-07/src/00b_service_principal.py
```

O endpoint precisa ler o prompt do Prompt Registry, que não entra nos `resources` (credenciais
automáticas) do serving. Este passo cria um **Service Principal**, guarda as credenciais OAuth dele
num **secret scope** (`aula_07`) e concede o mínimo: `USE CATALOG`; no schema, `USE SCHEMA`,
`EXECUTE`, `SELECT` e `CREATE FUNCTION` + `MANAGE` (o Prompt Registry exige esse trio até para
**ler** o prompt — num projeto real, isole os prompts num schema próprio); `CAN_USE` no endpoint
de Vector Search e `CAN_EDIT` no experimento de produção. Idempotente.

### Passo 1 — Desenvolver o agente (tools + RAG + observabilidade)

```bash
python aula-07/src/01_desenvolver_agente.py
```

**Observe**: o agente chama **UC Functions** (status do pedido, frete), recupera a política no
**Vector Search** (RAG) e cada requisição vira **uma trace** (via `autolog`), etiquetada com a
**versão do prompt** carregada do registry (`prompts:/...@champion`).

### Passo 2 — Versionar o prompt (MLflow Prompt Registry)

```bash
python aula-07/src/02_registry_versionamento.py
```

**Observe**: partindo da versão em produção (`champion`), registramos uma **candidata** imutável
(`challenger`), comparamos as duas frase a frase e demonstramos promoção e rollback — terminando
com a produção intacta. O agente (local ou no endpoint) resolve o alias a cada requisição, então
mover o `champion` muda o prompt usado **sem novo deploy** (cache de até 60 s).

### Passo 3 — Feedback humano

```bash
python aula-07/src/03_feedback_humano.py
```

**Observe**: o 👍/👎 do usuário é gravado com `mlflow.log_feedback` como *assessment* HUMAN
ligada à **trace** — sinal que alimenta a avaliação e o gate.

### Passo 4 — Gate de aprovação (testes unitários)

```bash
pytest aula-07/src/04_gate_aprovacao.py -v
```

O gate verifica **situações conhecidas** com três tipos de checagem: **regex** (garantia → "12 meses"
ou equivalente), **texto exato** (troca → "30 dias"; pedido 1002 → status real da tabela `pedidos`) e
um **LLM-as-a-judge** (scorer `Correctness`: reembolso → "7 dias úteis"). O agente roda com
`temperature=0` para respostas mais estáveis. *Exit code* != 0 **bloqueia o merge** no CI.

### Passo 5 — Servir o agente (deploy real)

```bash
python aula-07/src/05_serving_deploy.py
```

`log_model` (declarando os `resources`) no **experimento de produção** → registro no **Unity
Catalog** → `agents.deploy`. O endpoint roda como o **Service Principal** do Passo 0b
(`DATABRICKS_CLIENT_ID`/`SECRET` referenciados no secret scope) e lê o prompt `@champion` do
registry a cada requisição. **Observe**: o endpoint herda as credenciais dos recursos automaticamente e grava as traces em `/Users/<você>/agente_suporte_producao` — separado do
`experimento_aula_07` das demos. A resposta traz o `trace_id` em `custom_outputs`, e o feedback do
usuário (`registrar_feedback(trace_id, ...)`) é anexado a essa trace de produção.

> **Free Edition:** o `agents.deploy` mantém as versões anteriores servidas no endpoint, e o limite
> de concorrência provisionada do Free Edition não comporta várias. Por isso o Passo 5 remove o
> deploy anterior antes do novo (em produção, você manteria a versão anterior para rollback).

### Passo 6 — Orquestrar: o pipeline ponta a ponta

```bash
python aula-07/src/06_pipeline_llmops.py
```

**Observe**: registrar o prompt como `challenger` → **gate (pytest) rodando com o challenger**
(`PROMPT_ALIAS=challenger`) → promover a `champion` → servir → feedback, cada transição numa
**trilha de auditoria** (a governança que separa um protótipo de um sistema enterprise).

### (Demo) Gate verde/vermelho — o CI/CD reagindo aos dados

```bash
python aula-07/src/98_demo_gate_cicd.py
```

Remove o pedido **1002** da tabela `pedidos` para o gate **falhar** (*exit* != 0 → merge
bloqueado) e o recoloca para o gate voltar a **passar** — a mecânica exata da esteira de CI/CD,
demonstrada com dados reais do Unity Catalog.

### (Opcional) Limpeza — zerar a estrutura da aula

```bash
python aula-07/src/99_limpar_recursos.py
```

**Apaga** a estrutura atual da aula no catálogo/schema do `.env` (padrão `aula_07.default`):
endpoint de serving, modelo registrado, prompt, índice + endpoint de Vector Search, tabelas
(`pedidos`, `politicas`) e UC Functions. É **destrutivo** — use para recomeçar do zero (o
`00_setup` recria tudo). O experimento/traces do MLflow são preservados.

---

## 4. O notebook (a jornada interativa)

```bash
jupyter notebook aula-07/notebooks/exercicio_llmops.ipynb
```

Reproduz os passos célula a célula (ideal de rodar dentro de um workspace Databricks). A primeira
célula confere as credenciais.

---

## 5. A esteira de CI/CD (GitHub Actions)

O workflow [`.github/workflows/ci-cd-agente.yml`](../.github/workflows/ci-cd-agente.yml) tem **3 jobs encadeados**:

```
verificacoes  →  gate-aprovacao  →  registrar-deploy (só no main)
(código)         (pytest)            (registry + alias + Model Serving)
```

- **`verificacoes`** — compila o pacote e importa a config (código, sem segredos).
- **`gate-aprovacao`** — roda o `pytest` do Passo 4 contra o agente real; *exit code* != 0 **barra
  o merge**. Precisa dos segredos `DATABRICKS_HOST`/`DATABRICKS_TOKEN`; sem eles, os testes são
  **pulados** (a esteira segue verde).
- **`registrar-deploy`** — só no `main` e só se o gate passou; registra a versão, move o alias
  `champion` e faz o deploy (`agents.deploy`).

> O workflow dispara em `pull_request`/`push` que toquem `aula-07/**` (e também manualmente, via
> `workflow_dispatch`). Como o runner não tem o `.env`, o endpoint do LLM vem da variável de
> repositório `ENDPOINT_LLM_AGENTE` (padrão: `databricks-meta-llama-3-3-70b-instruct`).

---

## 6. O que você pratica nesta aula

Agente como **solução de software completa** (UC Functions · RAG por Vector Search · observabilidade
por `autolog` · feedback humano) autorado num **MLflow `ResponsesAgent`** · LLMOps × MLOps ·
versionamento do prompt no **Prompt Registry** com aliases champion/challenger (Unity Catalog) ·
**gate de aprovação** por testes unitários em CI/CD · **GitHub Actions** como esteira ·
**Model Serving** via `agents.deploy` · governança ponta a ponta (**Databricks Agent Framework**,
**Unity Catalog**).
