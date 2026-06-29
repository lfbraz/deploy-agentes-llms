"""
Exemplo 01 — Agente Reativo Simples (sem LLM)
=============================================

Objetivo didático
------------------
Mostrar, no código mais simples possível, a anatomia de um AGENTE:
percepção -> função do agente -> ação, dentro de um AMBIENTE.

Usamos o clássico "mundo do aspirador de pó" (vacuum world) de Russell & Norvig
(2021), aqui com QUATRO salas em linha (A, B, C, D). O agente é REATIVO: ele
decide a próxima ação olhando para a percepção atual (regras condição-ação),
sem planejar e sem LLM.

Repare na DIVISÃO DE RESPONSABILIDADES:
- O AMBIENTE é apenas o ESTADO do mundo (dados): quais salas existem, quais
  estão sujas e onde o aspirador está.
- O AGENTE tem os SENSORES (perceber) e os ATUADORES (agir): é ele que enxerga
  o ambiente e age sobre ele. A decisão fica na função do agente (decidir).

Para varrer o corredor inteiro (ir até a parede, bater e voltar limpando), o
agente guarda APENAS uma coisa: o SENTIDO em que está se movendo. É a única
"memória" dele — toda a inteligência continua em regras fixas.

Modelo PEAS deste agente
------------------------
- Performance (desempenho): número de salas limpas ao longo do tempo.
- Environment (ambiente): quatro salas em linha (A-B-C-D), cada uma Limpa ou Suja.
- Actuators (atuadores): mover para Esquerda, mover para Direita, Aspirar.
- Sensors (sensores): a sala atual, se ela está suja e se há parede ao lado.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1) O AMBIENTE (apenas o ESTADO do mundo em que o agente está)
# ---------------------------------------------------------------------------
@dataclass
class Ambiente:
    """Estado do mundo. 

    - salas: nome da sala -> True (suja) / False (limpa). A ordem de inserção
      define o corredor (ex.: A à esquerda, D à direita).
    - local: em qual sala o aspirador está.
    - desempenho: salas limpas até agora (medida de Performance do PEAS).
    """

    salas: dict[str, bool]
    local: str
    desempenho: int = 0


# ---------------------------------------------------------------------------
# 2) O AGENTE (sensores + decisão + atuadores)
# ---------------------------------------------------------------------------
class AspiradorReativo:
    """Agente reativo. Percebe o ambiente, decide e age sobre ele.

    A única memória é `sentido` (Esquerda/Direita): serve apenas para o
    aspirador saber para onde continuar varrendo. Ele NÃO planeja, NÃO sabe
    onde há sujeira longe dele e NÃO usa LLM.
    """

    def __init__(self, sentido: str = "Direita"):
        self.sentido = sentido

    # --- SENSORES: como o agente enxerga o ambiente ---
    def perceber(self, ambiente: Ambiente) -> dict:
        """Lê do ambiente só o que os sensores alcançam: a sala atual."""
        ordem = list(ambiente.salas.keys())
        idx = ordem.index(ambiente.local)
        return {
            "local": ambiente.local,
            "sujo": ambiente.salas[ambiente.local],
            "parede_esquerda": idx == 0,
            "parede_direita": idx == len(ordem) - 1,
        }

    # --- DECISÃO: a "função do agente" (regras condição-ação) ---
    def decidir(self, percepcao: dict) -> str:
        # Regra 1: se a sala onde estou está suja, aspiro. (prioridade máxima)
        if percepcao["sujo"]:
            return "Aspirar"

        # Regra 2: se bati numa parede, inverto o sentido da varredura.
        if percepcao["parede_direita"]:
            self.sentido = "Esquerda"
        elif percepcao["parede_esquerda"]:
            self.sentido = "Direita"

        # Regra 3: caso contrário, sigo na direção atual.
        return self.sentido

    # --- ATUADORES: como o agente age sobre o ambiente ---
    def agir(self, ambiente: Ambiente, acao: str) -> None:
        """Aplica a ação no mundo, alterando o ESTADO do ambiente."""
        ordem = list(ambiente.salas.keys())
        idx = ordem.index(ambiente.local)
        if acao == "Aspirar":
            if ambiente.salas[ambiente.local]:
                ambiente.salas[ambiente.local] = False
                ambiente.desempenho += 1
        elif acao == "Direita" and idx < len(ordem) - 1:
            ambiente.local = ordem[idx + 1]
        elif acao == "Esquerda" and idx > 0:
            ambiente.local = ordem[idx - 1]


# ---------------------------------------------------------------------------
# 3) APRESENTAÇÃO (funções auxiliares para visualizar/consultar o estado)
# ---------------------------------------------------------------------------
SETA = {"Direita": "→", "Esquerda": "←", "Aspirar": "↧"}


def tudo_limpo(ambiente: Ambiente) -> bool:
    return not any(ambiente.salas.values())


def salas_limpas(ambiente: Ambiente) -> int:
    return sum(1 for sujo in ambiente.salas.values() if not sujo)


def render(ambiente: Ambiente) -> str:
    """Desenho ASCII do corredor (alinhado): V = aspirador, ::: = sujeira."""
    ordem = list(ambiente.salas.keys())
    celula = "─────"
    topo = "┌" + "┬".join([celula] * len(ordem)) + "┐"
    rotulos = "│" + "│".join(f"  {s}  " for s in ordem) + "│"
    sujeira = "│" + "│".join(" ::: " if ambiente.salas[s] else "     " for s in ordem) + "│"
    robo = "│" + "│".join("  V  " if ambiente.local == s else "     " for s in ordem) + "│"
    base = "└" + "┴".join([celula] * len(ordem)) + "┘"
    return "\n".join([topo, rotulos, sujeira, robo, base])


def _limpar_tela() -> None:
    os.system("cls" if os.name == "nt" else "clear")


# ---------------------------------------------------------------------------
# 4) O LOOP PERCEPÇÃO-AÇÃO (conecta agente e ambiente no tempo)
# ---------------------------------------------------------------------------
def rodar_simulacao(passos: int = 30) -> None:
    # Cenário: todas as salas sujas, aspirador começa em B indo para a direita.
    # Assim ele varre até a parede (D), inverte e volta limpando o que falta (A).
    ambiente = Ambiente(
        salas={"A": True, "B": True, "C": True, "D": True},
        local="B",
    )
    aspirador = AspiradorReativo(sentido="Direita")

    historico: list[str] = ["0. Início: aspirador na sala B, sujeira em todas."]

    def desenhar(titulo: str) -> None:
        _limpar_tela()
        print("=" * 46)
        print("AGENTE REATIVO SIMPLES — Aspirador em 4 salas")
        print("=" * 46)
        print(render(ambiente))
        print("\nLegenda:  V = aspirador   |   ::: = sujeira")
        print(f"Salas limpas: {salas_limpas(ambiente)}/{len(ambiente.salas)}\n")
        print(titulo)
        print("Histórico de decisões:")
        for linha in historico:
            print("  " + linha)

    desenhar("Estado inicial:")

    for passo in range(1, passos + 1):
        percepcao = aspirador.perceber(ambiente)     # PERCEPÇÃO (sensores)
        acao = aspirador.decidir(percepcao)          # DECISÃO (regras condição-ação)
        aspirador.agir(ambiente, acao)               # AÇÃO (atuadores)

        estado = "suja" if percepcao["sujo"] else "limpa"
        if acao == "Aspirar":
            motivo = "está suja → Aspirar"
        elif percepcao["parede_direita"]:
            motivo = "parede à direita → inverte e vai p/ Esquerda"
        elif percepcao["parede_esquerda"]:
            motivo = "parede à esquerda → inverte e vai p/ Direita"
        else:
            motivo = f"limpa → segue {acao}"
        historico.append(f"{passo}. Sala {percepcao['local']} ({estado}): {motivo}  {SETA[acao]}")

        desenhar(f"Passo {passo}:")

        if tudo_limpo(ambiente):
            break

    print(f"\nDesempenho final (PEAS - Performance): {ambiente.desempenho} salas limpas.")
    print(
        "\nReflexão: este agente NÃO usa LLM. A 'inteligência' está em regras\n"
        "condição-ação fixas (classe AspiradorReativo). Nas próximas etapas,\n"
        "trocaremos essas regras rígidas por um LLM capaz de RACIOCINAR sobre\n"
        "percepções descritas em linguagem natural."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agente reativo simples — aspirador em 4 salas.")
    parser.add_argument("--passos", type=int, default=30,
                        help="limite de passos da simulação (padrão: 30).")
    args = parser.parse_args()

    rodar_simulacao(passos=args.passos)
