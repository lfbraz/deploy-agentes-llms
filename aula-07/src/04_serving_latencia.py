"""
Exemplo 04 — Deploy e serving: cache, cascata de modelos e latência (p50/p95)
=============================================================================

Versionado e aprovado no gate (exemplos 02-03), o agente é IMPLANTADO num endpoint de Model
Serving do Databricks. Servir bem é metade do LLMOps: a inferência de LLM é cara e lenta, e o
tempo de resposta varia muito com a entrada e o modelo escolhido. Duas alavancas arquiteturais
dominam a redução de latência/custo, e as exercitamos aqui contra **endpoints reais**:

  1. CACHE de respostas: prompts repetidos não pagam inferência duas vezes — um cache hit é
     ordens de grandeza mais rápido e custa zero.
  2. ROTEAMENTO / CASCATA de modelos: tente primeiro um endpoint PEQUENO e barato; só ESCALE
     para o GRANDE quando a pergunta for difícil. A maioria das perguntas é fácil.

Medimos a latência REAL (de rede) pela CAUDA (p95), não pela média — é o p95 que o usuário
sente e que dispara o SLA. Configure os dois endpoints no aula-07/.env:
`ENDPOINT_LLM_PEQUENO` (ex.: um Llama/Haiku) e `ENDPOINT_LLM_GRANDE` (ex.: um Claude Sonnet).
Requer DATABRICKS_HOST/TOKEN.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agente.mlflow_databricks import configurar_mlflow, credenciais_ok   # noqa: E402

ENDPOINT_PEQUENO = os.getenv("ENDPOINT_LLM_PEQUENO", "databricks-llama-4-maverick")
ENDPOINT_GRANDE = os.getenv("ENDPOINT_LLM_GRANDE", "databricks-claude-sonnet-4-5")
CUSTO_POR_TOKEN = {"pequeno": 2e-7, "grande": 3e-6}    # proxy de preço por token


def _cliente():
    configurar_mlflow()
    import mlflow.deployments
    return mlflow.deployments.get_deploy_client("databricks")


def consultar(cliente, endpoint: str, pergunta: str) -> tuple[str, float, int]:
    """Chama um endpoint de chat REAL e devolve (texto, latência_ms, tokens_estimados)."""
    t0 = time.perf_counter()
    resp = cliente.predict(endpoint=endpoint, inputs={
        "messages": [{"role": "user", "content": pergunta}],
        "max_tokens": 256, "temperature": 0,
    })
    latencia = (time.perf_counter() - t0) * 1000
    texto = resp["choices"][0]["message"]["content"]
    tokens = max(1, (len(pergunta) + len(texto)) // 4)
    return texto, latencia, tokens


def precisa_escalar(pergunta: str, resposta_pequena: str) -> bool:
    """Heurística de confiança: escala ao modelo grande se a pergunta é difícil/resposta fraca."""
    return (len(pergunta) >= 60 or len(resposta_pequena) < 20
            or "não sei" in resposta_pequena.lower())


@dataclass
class ServingEngine:
    """Camada de serving com cache + cascata de endpoints reais + telemetria de latência."""
    cliente: object = None
    _cache: dict[str, str] = field(default_factory=dict)
    latencias_ms: list[float] = field(default_factory=list)
    custo_total: float = 0.0
    hits: int = 0
    escaladas: int = 0

    def responder(self, pergunta: str) -> str:
        t0 = time.perf_counter()
        if pergunta in self._cache:                       # (1) CACHE exato
            self.hits += 1
            self.latencias_ms.append((time.perf_counter() - t0) * 1000)
            return self._cache[pergunta]

        # (2) CASCATA: tenta o pequeno; escala ao grande só se preciso.
        resposta, _, tok = consultar(self.cliente, ENDPOINT_PEQUENO, pergunta)
        self.custo_total += tok * CUSTO_POR_TOKEN["pequeno"]
        if precisa_escalar(pergunta, resposta):
            self.escaladas += 1
            resposta, _, tok = consultar(self.cliente, ENDPOINT_GRANDE, pergunta)
            self.custo_total += tok * CUSTO_POR_TOKEN["grande"]

        self._cache[pergunta] = resposta
        self.latencias_ms.append((time.perf_counter() - t0) * 1000)
        return resposta

    def painel(self) -> dict:
        lat = sorted(self.latencias_ms)
        n = len(lat)
        return {
            "requisicoes": n,
            "cache_hits": self.hits,
            "cache_hit_rate": round(self.hits / n, 2),
            "escaladas_p/_grande": self.escaladas,
            "latencia_p50_ms": round(lat[int(0.50 * (n - 1))]),
            "latencia_p95_ms": round(lat[int(0.95 * (n - 1))]),
            "latencia_max_ms": round(lat[-1]),
            "custo_total_usd": round(self.custo_total, 5),
        }


def main() -> int:
    print("=" * 70)
    print("DEPLOY E SERVING — cache + cascata de endpoints reais + latência p50/p95")
    print("=" * 70)

    if not credenciais_ok():
        print("\n⚠️  O serving consulta endpoints REAIS do Databricks. Defina DATABRICKS_HOST/")
        print("    TOKEN e os endpoints ENDPOINT_LLM_PEQUENO/ENDPOINT_LLM_GRANDE no .env.")
        return 1

    print(f"\n   endpoint pequeno: {ENDPOINT_PEQUENO}")
    print(f"   endpoint grande : {ENDPOINT_GRANDE}\n")

    # Tráfego: muitas perguntas fáceis e repetidas (cache), algumas longas/difíceis (escalam).
    faceis = ["Qual o horário de funcionamento?", "Vocês têm loja física?",
              "Qual o telefone do SAC?"]
    dificeis = [
        "Explique detalhadamente toda a política de devolução, reembolso e garantia estendida "
        "para produtos eletrônicos importados com nota fiscal de outro estado.",
    ]
    trafego = (faceis * 3) + dificeis      # 9 fáceis (com repetição -> cache) + 1 difícil

    engine = ServingEngine(cliente=_cliente())
    for pergunta in trafego:
        engine.responder(pergunta)

    print(f">> {len(trafego)} requisições processadas.\n")
    for chave, valor in engine.painel().items():
        print(f"   {chave:22}: {valor}")

    print(
        "\nLição: a maior parte do tráfego é fácil e repetido. CACHE elimina inferência\n"
        "redundante e a CASCATA reserva o endpoint caro para o que realmente precisa —\n"
        "derrubando p95 e custo juntos. Otimize sempre olhando a CAUDA (p95), não a média."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
