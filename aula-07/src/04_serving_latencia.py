# Databricks notebook source
# MAGIC %md
# MAGIC # Passo 4 — Servir o agente: cache, cascata e latência (p50/p95)
# MAGIC
# MAGIC Aprovado e promovido, o agente é implantado num endpoint de Model Serving. Servir bem é
# MAGIC metade do LLMOps: a inferência de LLM é cara e lenta, e o tempo de resposta varia muito com a
# MAGIC entrada e o modelo escolhido. Duas alavancas dominam a redução de latência/custo:
# MAGIC
# MAGIC - **cache**: prompts repetidos não pagam inferência duas vezes — um *hit* é ordens de grandeza
# MAGIC   mais rápido e custa zero;
# MAGIC - **cascata de modelos**: tente primeiro um endpoint pequeno e barato; só **escale** para o
# MAGIC   grande quando a pergunta for difícil. A maioria das perguntas é fácil.
# MAGIC
# MAGIC Medimos a latência pela **cauda (p95)**, não pela média. Configure os endpoints no
# MAGIC `aula-07/.env`: `ENDPOINT_LLM_PEQUENO` e `ENDPOINT_LLM_GRANDE`.

# COMMAND ----------

import os
import sys
import time
from dataclasses import dataclass, field

try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()
while _dir != os.path.dirname(_dir):
    if os.path.isdir(os.path.join(_dir, "agente")):
        sys.path.insert(0, _dir)
        break
    _dir = os.path.dirname(_dir)

import mlflow.deployments

from agente.mlflow_databricks import configurar_mlflow, credenciais_ok

if not credenciais_ok():
    raise SystemExit("Configure DATABRICKS_HOST/TOKEN no aula-07/.env (veja o .env.example).")

configurar_mlflow()
ENDPOINT_PEQUENO = os.getenv("ENDPOINT_LLM_PEQUENO", "databricks-llama-4-maverick")
ENDPOINT_GRANDE = os.getenv("ENDPOINT_LLM_GRANDE", "databricks-claude-sonnet-4-5")
CUSTO_POR_TOKEN = {"pequeno": 2e-7, "grande": 3e-6}
cliente = mlflow.deployments.get_deploy_client("databricks")
print("endpoint pequeno:", ENDPOINT_PEQUENO)
print("endpoint grande :", ENDPOINT_GRANDE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Consultar um endpoint e decidir a cascata

# COMMAND ----------

def consultar(endpoint, pergunta):
    """Chama um endpoint de chat e devolve (texto, tokens_estimados)."""
    resp = cliente.predict(endpoint=endpoint, inputs={
        "messages": [{"role": "user", "content": pergunta}],
        "max_tokens": 256, "temperature": 0})
    texto = resp["choices"][0]["message"]["content"]
    tokens = max(1, (len(pergunta) + len(texto)) // 4)
    return texto, tokens

def precisa_escalar(pergunta, resposta_pequena):
    """Confiança heurística: escala ao grande se a pergunta é difícil/resposta fraca."""
    return (len(pergunta) >= 60 or len(resposta_pequena) < 20
            or "não sei" in resposta_pequena.lower())

# COMMAND ----------

# MAGIC %md
# MAGIC ## A camada de serving: cache + cascata + telemetria

# COMMAND ----------

@dataclass
class ServingEngine:
    cache: dict = field(default_factory=dict)
    latencias_ms: list = field(default_factory=list)
    custo_total: float = 0.0
    hits: int = 0
    escaladas: int = 0

    def responder(self, pergunta):
        t0 = time.perf_counter()
        if pergunta in self.cache:                        # (1) cache exato
            self.hits += 1
            self.latencias_ms.append((time.perf_counter() - t0) * 1000)
            return self.cache[pergunta]
        resposta, tok = consultar(ENDPOINT_PEQUENO, pergunta)   # (2) tenta o pequeno
        self.custo_total += tok * CUSTO_POR_TOKEN["pequeno"]
        if precisa_escalar(pergunta, resposta):                 # escala ao grande se preciso
            self.escaladas += 1
            resposta, tok = consultar(ENDPOINT_GRANDE, pergunta)
            self.custo_total += tok * CUSTO_POR_TOKEN["grande"]
        self.cache[pergunta] = resposta
        self.latencias_ms.append((time.perf_counter() - t0) * 1000)
        return resposta

    def painel(self):
        lat = sorted(self.latencias_ms)
        n = len(lat)
        return {"requisicoes": n, "cache_hits": self.hits,
                "cache_hit_rate": round(self.hits / n, 2),
                "escaladas_p/_grande": self.escaladas,
                "latencia_p50_ms": round(lat[int(0.50 * (n - 1))]),
                "latencia_p95_ms": round(lat[int(0.95 * (n - 1))]),
                "latencia_max_ms": round(lat[-1]),
                "custo_total_usd": round(self.custo_total, 5)}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tráfego: muitas perguntas fáceis e repetidas, algumas difíceis

# COMMAND ----------

faceis = ["Qual o horário de funcionamento?", "Vocês têm loja física?", "Qual o telefone do SAC?"]
dificeis = ["Explique detalhadamente toda a política de devolução, reembolso e garantia estendida "
            "para produtos eletrônicos importados com nota fiscal de outro estado."]
trafego = (faceis * 3) + dificeis      # 9 fáceis (com repetição -> cache) + 1 difícil

engine = ServingEngine()
for pergunta in trafego:
    engine.responder(pergunta)

print(f">> {len(trafego)} requisições processadas.\n")
for chave, valor in engine.painel().items():
    print(f"   {chave:22}: {valor}")

# COMMAND ----------

# MAGIC %md
# MAGIC A maior parte do tráfego é fácil e repetido. O **cache** elimina inferência redundante e a
# MAGIC **cascata** reserva o endpoint caro para o que realmente precisa — derrubando p95 e custo
# MAGIC juntos. Otimize sempre olhando a **cauda (p95)**, não a média.
