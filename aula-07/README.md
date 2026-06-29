# Aula 07 — LLMOps: a jornada completa de um agente em produção

Nesta aula construímos um **agente de suporte de e-commerce** e o levamos por toda a
disciplina de **LLMOps** — o ciclo de vida operacional que transforma um protótipo num produto:

```
DESENVOLVER → REGISTRAR → TESTAR (CI/CD) → SERVIR (deploy) → MONITORAR → ORQUESTRAR
```

O agente é desenvolvido na interface **`ResponsesAgent` do MLflow**, o padrão recomendado pelo
**Databricks Agent Framework**. O "cérebro" é um modelo Claude servido pelo Databricks, a base vetorial usa
embeddings do serviço AI Search da Databricks, o registry é o **MLflow Prompt Registry** no **Unity Catalog**, o
serving consulta **endpoints reais** e a observabilidade vai para o **MLflow Tracing** do
workspace.

> Sem `DATABRICKS_HOST`/`DATABRICKS_TOKEN`, cada exemplo exibe os pré-requisitos e encerra
> (as únicas checagens que rodam sem credencial são os **guardrails**, que são código).

---

## 1. O agente (`agente/`) — a peça central

O agente é uma **solução de software completa**, reunindo os pilares das aulas anteriores:

| Arquivo | Pilar (aula) | O que faz |
|---------|--------------|-----------|
| [`config.py`](agente/config.py) | versionamento | a "tripla" imutável: `system_prompt` + `endpoint` + geração |
| [`base_vetorial.py`](agente/base_vetorial.py) | RAG (Aula 4) | índice das políticas com **DatabricksEmbeddings** |
| [`ferramentas.py`](agente/ferramentas.py) | tool use (Aulas 3-4) | `status_pedido`, `frete`, `buscar_politica` (`@tool`) |
| [`guardrails.py`](agente/guardrails.py) | segurança (Aula 5) | filtros de entrada (jailbreak/PII) e saída (vazamento) |
| [`agente_suporte.py`](agente/agente_suporte.py) | observabilidade (Aula 6) | `ResponsesAgent` (ChatDatabricks + loop de tool use) + trace/feedback |
| [`mlflow_databricks.py`](agente/mlflow_databricks.py) | infra | pré-requisitos, tracking/registry no Unity Catalog |
| [`serving_model.py`](agente/serving_model.py) | deploy | entrypoint *models-from-code* p/ `log_model` / `agents.deploy` |

---

## 2. Pré-requisitos e configuração (obrigatório)

1. **Python 3.10+** e as dependências (a partir da raiz do repositório):

   ```bash
   pip install -r requirements.txt        # inclui mlflow e databricks-langchain
   ```

2. **Credenciais do Databricks** — copie o template e preencha:

   ```bash
   cp aula-07/.env.example aula-07/.env
   ```

   No `aula-07/.env`, defina:
   - `DATABRICKS_HOST` e `DATABRICKS_TOKEN` (workspace + Personal Access Token);
   - `ENDPOINT_LLM_AGENTE` — um endpoint que suporte **tool calling** (ex.:
     `databricks-claude-sonnet-4-5`). **Não** use o `databricks-gpt-oss-20b` (não faz function calling);
   - `ENDPOINT_EMBEDDINGS` — endpoint de embeddings para o RAG;
   - `ENDPOINT_LLM_PEQUENO` / `ENDPOINT_LLM_GRANDE` — usados na cascata do serving (passo 4);
   - `MODELO_UC` — `catalogo.schema.objeto` para o registry/deploy (padrão `main.default.agente_suporte`).

> O `.env` (com suas credenciais) **nunca** é versionado — ele está no `.gitignore`.

---

## 3. Passo a passo — rodando a jornada

Execute os exemplos **na ordem** (a partir da raiz do repositório). Cada um é uma etapa do
ciclo de vida do agente.

### Passo 1 — Desenvolver o agente

```bash
python aula-07/src/01_desenvolver_agente.py
```

Roda o agente em uma conversa de exemplo. **Observe**: o agente *chama ferramentas* (status do
pedido, frete), *recupera a política* da base vetorial (RAG), tem a entrada *bloqueada* num
jailbreak e a *PII mascarada*, e cada requisição vira uma **trace** no MLflow (latência, tokens,
ferramentas). No fim, registra um **feedback** 👍.

### Passo 2 — Registrar e versionar (MLflow Prompt Registry / Unity Catalog)

```bash
python aula-07/src/02_registry_versionamento.py
```

**Observe**: cada versão do prompt é **imutável** no Prompt Registry, e **aliases móveis**
(`champion`, `challenger`) decidem o que vai a produção.

### Passo 3 — Testar: o gate de aprovação (instruções que devem ser respeitadas)

```bash
python aula-07/src/03_gate_regressao_cicd.py        # modo demo
python aula-07/src/03_gate_regressao_cicd.py --ci   # modo da esteira
```

O agente tem **instruções obrigatórias** (citar o prazo de troca, informar a garantia, usar a
ferramenta de pedido, recusar *jailbreak*). Quando alguém **altera as instruções**, o gate
valida se o conteúdo retornado **continua respeitando cada regra**.

- **modo demo**: um candidato com *system prompt* alterado (passa a ignorar ferramentas/política)
  → o conteúdo fica **inválido** e o gate **barra** (no GitHub Actions, *exit code* **1** bloqueia
  o merge do PR).
- **modo `--ci`**: valida a versão atual contra o mínimo de conformidade → **válido** (exit **0**).

> Verifique o código de saída com `echo $?` — é ele que barra (ou libera) o deploy.

### Passo 4 — Servir (deploy): cache, cascata e latência

```bash
python aula-07/src/04_serving_latencia.py
```

Consulta **dois endpoints reais** (`ENDPOINT_LLM_PEQUENO` e `ENDPOINT_LLM_GRANDE`). **Observe**:
o **cache** elimina inferência repetida e a **cascata** (tenta o pequeno; escala ao grande só
quando a pergunta é difícil) derruba custo. A latência é **real** e medida pela **cauda (p95)**.

### Passo 5 — Monitorar (p95, custo, sucesso, conformidade)

```bash
python aula-07/src/05_monitoramento.py
```

Roda o agente sobre um lote representativo e monta o painel a partir da telemetria. **Observe**:
além de p95, custo, sucesso e guardrail, mede a **conformidade** (as respostas seguem respeitando
as instruções do Passo 3?). Limites de **SLA** disparam alertas — e uma queda de conformidade vira
um caso novo no *golden set*, **realimentando o gate do Passo 3**.

### Passo 6 — Orquestrar: o pipeline ponta a ponta

```bash
python aula-07/src/06_pipeline_llmops.py
```

**Observe**: os passos se **encaixam** — desenvolver → registrar → gate → promover → servir →
monitorar — e cada transição é registrada numa **trilha de auditoria** (a governança que separa
um protótipo de um sistema enterprise).

---

## 4. O notebook (a jornada interativa)

```bash
jupyter notebook aula-07/notebooks/exercicio_llmops.ipynb
```

Reproduz os 6 passos célula a célula (ideal de rodar dentro de um workspace Databricks ou
localmente com o `.env` configurado). A primeira célula confere as credenciais.

---

## 5. A esteira de CI/CD (GitHub Actions)

O workflow [`.github/workflows/ci-cd-agente.yml`](../.github/workflows/ci-cd-agente.yml) tem
**3 jobs encadeados**:

```
testes  →  gate-regressao  →  registrar-deploy (só no main)
(código)   (agente real)      (registry + alias + Model Serving)
```

- **`testes`** — valida os **guardrails** e o import do pacote (código, sem segredos).
- **`gate-regressao`** — roda o Passo 3 (`--ci`) contra o agente real; *exit code* != 0 **barra
  o merge**. Precisa dos segredos `DATABRICKS_HOST`/`DATABRICKS_TOKEN` (*Settings → Secrets and
  variables → Actions*); sem eles, o passo é **pulado com aviso** (a esteira segue verde).
- **`registrar-deploy`** — só no `main` e só se o gate passou; registra a versão, move o alias
  `champion` e faz o deploy (`agents.deploy`).

---

## 6. O que você pratica nesta aula

Agente como **solução de software completa** (tool use · RAG · guardrails · observabilidade ·
feedback) autorado num **MLflow `ResponsesAgent`** · LLMOps × MLOps · versionamento do prompt no
**Prompt Registry** com aliases champion/challenger (Unity Catalog) · **gate de aprovação** que
valida conformidade às instruções obrigatórias em CI/CD · **GitHub Actions** como esteira ·
**Model Serving** e otimização de latência (cache, cascata, p95) · **monitoramento** (p95, custo,
sucesso, conformidade, SLA) · governança ponta a ponta (**Databricks Agent Framework**, **Unity
Catalog**).

## 7. Desafios (opcional)

1. No Passo 2, mantenha um alias `previous` além de `champion` e faça o *rollback* apontar para
   ele em O(1).
2. No Passo 3, troque os asserts semânticos por *scorers* do `mlflow.genai.evaluate()` (juízes
   LLM da Aula 6) e compare os vereditos.
3. No Passo 4, troque o cache exato por um **cache semântico** (embeddings) e meça o ganho de
   *hit rate* em perguntas reformuladas.
4. No Passo 5, faça uma queda de **conformidade** disparar a criação **automática** de um novo
   caso no golden set — fechando o ciclo monitoramento → gate.
