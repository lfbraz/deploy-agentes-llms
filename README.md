# Deploy de Agentes com LLMs

Material prático da disciplina **Deploy de Agentes com LLMs** (Pós-Graduação em Machine Learning —
FIAP/POSTECH), por **Luiz Fernando Braz**.

A ideia do curso é simples: começar com o **agente mais básico possível** (um aspirador de pó que
segue regras fixas) e, aula após aula, ir acrescentando peças até chegar a um **agente de verdade
rodando em produção**, com testes, versionamento, monitoramento e deploy automático. A última aula
olha para frente: **vários agentes trabalhando juntos**.

```
Aula 01 → Aula 02 → Aula 03 → Aula 04 → Aula 05 → Aula 06 → Aula 07 → Aula 08
O que é    Falar     Encadear   Memória   Segurança  Medir e    Colocar em  Vários
um agente  bem com   passos e   e ferra-  e limites  observar   produção    agentes
           o LLM     decidir    mentas                          (LLMOps)    juntos
```

> **Dica:** siga as aulas **em ordem**. Cada uma reaproveita ideias da anterior.

---

## Mapa das aulas

| Aula | Tema | Onde está o código | Precisa de quê? |
|------|------|--------------------|-----------------|
| [01](aula-01/) | O que é um agente | [`aula-01/src/`](aula-01/src/) (scripts Python) | Nada (chave da Anthropic é opcional) |
| [02](aula-02/) | Engenharia de prompt | [`exercicio_prompt_react.ipynb`](aula-02/notebooks/exercicio_prompt_react.ipynb) | Chave da Anthropic |
| [03](aula-03/) | Chains e loops de decisão | [`exercicio_chains_loops.ipynb`](aula-03/notebooks/exercicio_chains_loops.ipynb) | Chave da Anthropic |
| [04](aula-04/) | RAG, bancos vetoriais e ferramentas | [`exercicio_rag_tools.ipynb`](aula-04/notebooks/exercicio_rag_tools.ipynb) | Chave da Anthropic + internet |
| [05](aula-05/) | Guardrails e rate limits | [`guardrails_databricks.ipynb`](aula-05/notebooks/guardrails_databricks.ipynb), [`rate_limits_databricks.ipynb`](aula-05/notebooks/rate_limits_databricks.ipynb) | Workspace Databricks |
| [06](aula-06/) | Avaliação e observabilidade | [`exercicio_eval_observabilidade.ipynb`](aula-06/notebooks/exercicio_eval_observabilidade.ipynb), [`tracing_mlflow.ipynb`](aula-06/notebooks/tracing_mlflow.ipynb) | Workspace Databricks (o de *tracing* roda sem nada) |
| [07](aula-07/) | LLMOps: agente em produção | [`aula-07/`](aula-07/) — tem [README próprio](aula-07/README.md) | Workspace Databricks |
| [08](aula-08/) | Sistemas multi-agente | [`exercicio_topicos_avancados.ipynb`](aula-08/notebooks/exercicio_topicos_avancados.ipynb) | Chave da Anthropic |

---

## Aula 01 — O que é um agente?

**Onde está:** [`aula-01/src/`](aula-01/src/)

Usamos um exemplo clássico: um **aspirador de pó** que precisa limpar quatro salas (A, B, C e D).
O mesmo problema é resolvido de quatro jeitos diferentes, do mais simples ao mais "inteligente":

| Arquivo | O que mostra |
|---------|--------------|
| [`01_agente_reativo_simples.py`](aula-01/src/01_agente_reativo_simples.py) | Um agente **sem LLM**, que só segue regras do tipo "se a sala está suja, aspire". |
| [`02_agente_llm_motor.py`](aula-01/src/02_agente_llm_motor.py) | O mesmo agente, mas agora quem **decide** é um LLM — ele vira o "cérebro". |
| [`03_orquestracao_chain_vs_agente.py`](aula-01/src/03_orquestracao_chain_vs_agente.py) | **Workflow:** o LLM faz um plano uma única vez e o código executa esse plano. |
| [`04_agente_react_ferramentas.py`](aula-01/src/04_agente_react_ferramentas.py) | **Agente ReAct:** o LLM decide passo a passo, usando **ferramentas**, e se adapta ao que vê. |

**O que você vai aprender:** a diferença entre seguir regras, seguir um plano fixo e decidir sozinho
— e por que nem sempre vale a pena usar um agente quando um fluxo simples resolve.

> Funciona **sem chave de API**: nesse caso, os exemplos usam um "LLM simulado".

---

## Aula 02 — Engenharia de prompt na prática

**Onde está:** [`aula-02/notebooks/exercicio_prompt_react.ipynb`](aula-02/notebooks/exercicio_prompt_react.ipynb)

Partimos de **prompts ruins** e vamos melhorando aos poucos, vendo como cada técnica muda a resposta
do modelo:

1. Ser claro e específico;
2. Definir papel e tom (mensagem de *system*);
3. Pedir respostas em formato fixo (JSON / *structured output*);
4. Dar exemplos (*few-shot*);
5. Pedir raciocínio passo a passo (*Chain-of-Thought*);
6. Evitar que o modelo invente coisas (alucinação);
7. **Bônus:** um agente **ReAct** que pesquisa na web.

**O que você vai aprender:** como escrever prompts que dão respostas previsíveis e úteis.

---

## Aula 03 — Chains e loops de decisão

**Onde está:** [`aula-03/notebooks/exercicio_chains_loops.ipynb`](aula-03/notebooks/exercicio_chains_loops.ipynb)

Comparamos duas formas de montar uma aplicação com LLM, num exemplo de consulta de estoque e preços:

- **Chain:** os passos são sempre os mesmos e na mesma ordem (usando LangChain).
- **Agente:** o modelo escolhe, a cada volta, qual ação tomar.

Depois adicionamos **logs** para enxergar o que o agente fez e um **ciclo de autocorreção**, em que o
agente aprende com o próprio erro.

**O que você vai aprender:** quando usar uma chain (barata e previsível) e quando usar um agente
(flexível, mas mais caro).

---

## Aula 04 — Memória e ferramentas (RAG e APIs)

**Onde está:** [`aula-04/notebooks/exercicio_rag_tools.ipynb`](aula-04/notebooks/exercicio_rag_tools.ipynb)

Damos ao agente **memória** e **mãos**:

1. **Embeddings:** transformar texto em números para buscar por **significado**, não por palavra exata;
2. **Índice vetorial (FAISS):** encontrar rápido os textos mais parecidos;
3. **RAG:** buscar informação relevante e entregar ao LLM antes de ele responder;
4. **Tool use:** o próprio LLM decide chamar uma **API externa real** (cotação de moedas).

**O que você vai aprender:** como o agente consulta conhecimento que não estava no treino e como ele
interage com sistemas de fora.

---

## Aula 05 — Segurança: guardrails e rate limits

**Onde está:** [`aula-05/notebooks/`](aula-05/notebooks/)

Usamos o **Databricks AI Gateway** com dois endpoints que servem o mesmo modelo — um **sem** proteção
e outro **com** proteção — e comparamos os dois lado a lado.

| Notebook | O que mostra |
|----------|--------------|
| [`guardrails_databricks.ipynb`](aula-05/notebooks/guardrails_databricks.ipynb) | Bloqueio de **dados pessoais**, **conteúdo nocivo**, tentativas de **jailbreak** e **alucinação**. |
| [`rate_limits_databricks.ipynb`](aula-05/notebooks/rate_limits_databricks.ipynb) | Limite de requisições por minuto: depois de X chamadas, o endpoint responde "espere um pouco" (HTTP 429). |

**O que você vai aprender:** como proteger o agente (e a sua conta) sem mudar o código da aplicação.

> Dá para fazer tudo no **Databricks Free Edition**, gratuito:
> [databricks.com/learn/free-edition](https://www.databricks.com/learn/free-edition).

---

## Aula 06 — Avaliação e observabilidade

**Onde está:** [`aula-06/notebooks/`](aula-06/notebooks/)

Duas perguntas que todo agente em produção precisa responder: **"a resposta foi boa?"** e **"o que
exatamente o agente fez?"**

| Notebook | O que mostra |
|----------|--------------|
| [`exercicio_eval_observabilidade.ipynb`](aula-06/notebooks/exercicio_eval_observabilidade.ipynb) | Os **scorers do MLflow**: juízes prontos, regras escritas em português, juízes sob medida e checagens em código puro. |
| [`tracing_mlflow.ipynb`](aula-06/notebooks/tracing_mlflow.ipynb) | **Tracing:** cada execução vira uma "árvore" de passos com tempo, tokens e custo. Mostra também como montar métricas e painéis. |

**O que você vai aprender:** como medir a qualidade das respostas e como investigar o que deu errado.

> O notebook de *tracing* roda **sem nenhuma chave** (modo offline).

---

## Aula 07 — LLMOps: o agente em produção

**Onde está:** [`aula-07/`](aula-07/) — **leia o [README da Aula 07](aula-07/README.md)**, que tem o passo a passo completo.

Juntamos tudo num **agente de suporte de e-commerce** e o levamos por todo o ciclo de vida de um
produto:

```
SETUP → DESENVOLVER → VERSIONAR → FEEDBACK → GATE → SERVIR → CI/CD
```

| Onde | O que tem |
|------|-----------|
| [`agente/`](aula-07/agente/) | O código do agente (ferramentas, RAG, configuração). |
| [`src/`](aula-07/src/) | Scripts numerados `00` a `06`, um para cada etapa do ciclo (mais `98` e `99`, demo do gate e limpeza). |
| [`notebooks/exercicio_llmops.ipynb`](aula-07/notebooks/exercicio_llmops.ipynb) | A mesma jornada em formato de notebook, para rodar no Databricks. |
| [`.github/workflows/ci-cd-agente.yml`](.github/workflows/ci-cd-agente.yml) | A esteira de **CI/CD** no GitHub Actions: os testes do agente barram o merge se algo quebrar. |

**O que você vai aprender:** versionar prompts, coletar feedback do usuário, testar o agente
automaticamente e publicá-lo num endpoint de verdade.

---

## Aula 08 — Tópicos avançados: sistemas multi-agente

**Onde está:** [`aula-08/notebooks/exercicio_topicos_avancados.ipynb`](aula-08/notebooks/exercicio_topicos_avancados.ipynb)

Na aula de encerramento, em vez de um único agente fazendo tudo, montamos uma **equipe de agentes**.
O cenário é a central de atendimento de um banco, com três **especialistas**, cada um com sua própria
base de conhecimento:

- **Cartão de Crédito**;
- **Conta Corrente**;
- **Help Desk (TI)**.

O mesmo problema é resolvido de dois jeitos:

| Arquitetura | Como funciona |
|-------------|---------------|
| **Subagentes** | Um agente **coordenador** (a "recepção") trata cada especialista como uma ferramenta e decide sozinho quem chamar — inclusive mais de um, se a dúvida misturar assuntos. |
| **Router** | Um **classificador** quebra a mensagem em perguntas menores, envia cada uma ao especialista certo **ao mesmo tempo** e, no fim, junta tudo numa **resposta única** (com LangGraph). |

**O que você vai aprender:** como dividir o trabalho entre vários agentes e quando vale mais a pena
deixar o LLM decidir (subagentes, mais flexível) ou seguir um fluxo previsível (router, mais fácil
de controlar e escalar).

---

## Como preparar o ambiente

1. **Python 3.12.**

2. **Crie um ambiente virtual e instale as dependências** (na raiz do repositório):

   ```bash
   python3.12 -m venv .venv
   ```

   ```bash
   source .venv/bin/activate
   ```

   ```bash
   pip install -r requirements.txt
   ```

   A instalação leva alguns minutos: são muitas bibliotecas e o `pip` precisa conferir a
   compatibilidade entre elas.

   > Se aparecer `Could not find a version that satisfies the requirement databricks-connect`,
   > o ambiente foi criado com outra versão do Python. Confira com `python --version`, apague a
   > pasta `.venv` e crie de novo com `python3.12`.

3. **Configure as chaves de cada aula.** As pastas `aula-02/` a `aula-08/` têm um arquivo
   `.env.example` explicando o que preencher (a Aula 01 usa o `.env.example` da raiz). Copie para
   `.env` e complete. Por exemplo, para a Aula 02:

   ```bash
   cp aula-02/.env.example aula-02/.env
   ```

   - **Aula 01:** a chave da **Anthropic** (`ANTHROPIC_API_KEY`) é opcional.
   - **Aulas 02 a 04:** precisam de uma chave da **Anthropic** (`ANTHROPIC_API_KEY`).
   - **Aulas 05 a 07:** precisam do endereço e de um token de um **workspace Databricks**
     (`DATABRICKS_HOST` e `DATABRICKS_TOKEN`).
   - **Aula 08:** volta a usar a chave da **Anthropic** (`ANTHROPIC_API_KEY`).

4. **Abra os notebooks** no VS Code, no Jupyter ou direto no Databricks.

> O arquivo `.env` guarda suas credenciais e **nunca** vai para o GitHub (ele está no `.gitignore`).
> Só o `.env.example`, sem segredos, é versionado.
