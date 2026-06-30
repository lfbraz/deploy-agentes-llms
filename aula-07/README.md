# Aula 07 — LLMOps: a jornada completa de um agente em produção

Nesta aula construímos um **agente de suporte de e-commerce** sobre os primitivos **nativos do
Databricks Agent Framework** e o levamos por toda a disciplina de **LLMOps** — o ciclo de vida
operacional que transforma um protótipo num produto:

```
SETUP → DESENVOLVER → VERSIONAR → FEEDBACK → GATE → SERVIR → ORQUESTRAR (CI/CD)
```

O que demonstramos, ponta a ponta:

1. **Agente com tools** — ferramentas como **UC Functions** (Unity Catalog) via `UCFunctionToolkit`;
2. **Observabilidade** — `mlflow.langchain.autolog()` (cada chamada vira *span* numa trace);
3. **RAG com base vetorial** — índice de **Databricks AI Search (Vector Search)** + `VectorSearchRetrieverTool`;
4. **Feedback humano** — `mlflow.log_feedback` anexa o 👍/👎 do usuário à trace;
5. **Gate de aprovação** — **testes unitários** (pytest) do comportamento do agente;
6. **Prompt registry** — versionamento do prompt no **MLflow Prompt Registry** (Unity Catalog) com aliases;
7. **Serving endpoint** — `log_model` (com `resources`) → registro no UC → `agents.deploy`;
8. **CI/CD** — **GitHub Actions** com o gate de aprovação barrando o merge.

> Os **guardrails** de produção e os **painéis de monitoramento** são configurados na própria UI
> do workspace (AI Gateway / Guardrails, *Inference Tables*, dashboards). No código, um teste de
> *jailbreak* recusado entra como caso do **gate de aprovação**.

> Sem `DATABRICKS_HOST`/`DATABRICKS_TOKEN`, cada exemplo exibe os pré-requisitos e encerra. A Aula 07
> roda **sempre** contra o Databricks (não há modo offline).

---

## 1. O agente (`agente/`) — a peça central

O agente é uma **solução de software completa**, reunindo os pilares das aulas anteriores sobre
recursos governados:

| Arquivo | Pilar (aula) | O que faz |
|---------|--------------|-----------|
| [`config.py`](agente/config.py) | versionamento/infra | a "tripla" (prompt+endpoint+geração) + nomes UC + endpoints |
| [`ferramentas.py`](agente/ferramentas.py) | tool use (Aulas 3-4) | funções → **UC Functions** + `UCFunctionToolkit` |
| [`base_vetorial.py`](agente/base_vetorial.py) | RAG (Aula 4) | índice de **Vector Search** + `VectorSearchRetrieverTool` |
| [`agente_suporte.py`](agente/agente_suporte.py) | observabilidade (Aula 6) | `ResponsesAgent` (tools + RAG) + `autolog` + feedback |
| [`mlflow_databricks.py`](agente/mlflow_databricks.py) | infra | pré-requisitos, tracking/registry no Unity Catalog |
| [`serving_model.py`](agente/serving_model.py) | deploy | entrypoint *models-from-code* p/ `log_model` / `agents.deploy` |

---

## 2. Pré-requisitos e configuração (obrigatório)

1. **Python 3.10+** e as dependências (a partir da raiz do repositório):

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
   - `UC_CATALOG`/`UC_SCHEMA` — catálogo.schema **com permissão de escrita** (padrão `main.default`);
   - `ENDPOINT_LLM_AGENTE` — endpoint que suporte **tool calling** (ex.: `databricks-claude-sonnet-4-5`).
     **Não** use o `databricks-gpt-oss-20b` (não faz function calling);
   - `ENDPOINT_EMBEDDINGS` — endpoint de embeddings do índice de Vector Search;
   - `ENDPOINT_VECTOR_SEARCH` — nome do endpoint (compute) de Vector Search.

> O `.env` (com suas credenciais) **nunca** é versionado — ele está no `.gitignore`.

---

## 3. Passo a passo — rodando a jornada

Execute **na ordem** (a partir da raiz do repositório). Cada um é uma etapa do ciclo de vida.

### Passo 0 — Provisionar os recursos (UC Functions + Vector Search)

```bash
python aula-07/src/00_setup_recursos.py
```

Idempotente: registra as **UC Functions** (tools) e cria a tabela de políticas + o **índice de
Vector Search** (RAG). Roda **dentro de um workspace Databricks** (precisa da sessão `spark`).

### Passo 1 — Desenvolver o agente (tools + RAG + observabilidade)

```bash
python aula-07/src/01_desenvolver_agente.py
```

**Observe**: o agente chama **UC Functions** (status do pedido, frete), recupera a política no
**Vector Search** (RAG) e cada requisição vira uma **trace** (via `autolog`).

### Passo 2 — Versionar o prompt (MLflow Prompt Registry)

```bash
python aula-07/src/02_registry_versionamento.py
```

**Observe**: cada versão do prompt (com `{{variáveis}}`) é **imutável**, e **aliases móveis**
(`champion`, `challenger`) decidem o que vai a produção.

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

O gate verifica **situações conhecidas** (garantia → "12 meses", troca → "30 dias", pedido →
UC Function) e se um **jailbreak é recusado**. *Exit code* != 0 **bloqueia o merge** no CI.

### Passo 5 — Servir o agente (deploy real)

```bash
python aula-07/src/05_serving_deploy.py
```

`log_model` (declarando os `resources`) → registro no **Unity Catalog** → `agents.deploy`.
**Observe**: o endpoint herda as credenciais dos recursos automaticamente; a *review app* e a
*inference table* nascem junto.

### Passo 6 — Orquestrar: o pipeline ponta a ponta

```bash
python aula-07/src/06_pipeline_llmops.py
```

**Observe**: registrar prompt → **gate (pytest)** → promover alias → servir → feedback, cada
transição numa **trilha de auditoria** (a governança que separa um protótipo de um sistema enterprise).

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

> Os gatilhos automáticos (`pull_request`/`push`) estão **temporariamente comentados** no workflow
> durante a adaptação dos exemplos — restam apenas execuções manuais (`workflow_dispatch`).

---

## 6. O que você pratica nesta aula

Agente como **solução de software completa** (UC Functions · RAG por Vector Search · observabilidade
por `autolog` · feedback humano) autorado num **MLflow `ResponsesAgent`** · LLMOps × MLOps ·
versionamento do prompt no **Prompt Registry** com aliases champion/challenger (Unity Catalog) ·
**gate de aprovação** por testes unitários em CI/CD · **GitHub Actions** como esteira ·
**Model Serving** via `agents.deploy` · governança ponta a ponta (**Databricks Agent Framework**,
**Unity Catalog**).
