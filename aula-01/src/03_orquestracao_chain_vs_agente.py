"""
Exemplo 03 — Orquestração de Workflow: PLANEJAR e depois EXECUTAR
================================================================

Objetivo didático
------------------
Continuamos no mundo do aspirador (Exemplos 01 e 02), mas mudamos a forma de
orquestrar. Aqui o fluxo é um WORKFLOW de "planejar e executar" (Anthropic, 2024):

1) PLANEJAR (uma chamada ao LLM): dado o estado das salas, o LLM devolve um PLANO
   — a ordem das salas a limpar (ex.: B, C, D, A).
2) EXECUTAR (código determinístico): o nosso código percorre o plano na ordem,
   movendo o aspirador e aspirando. QUEM controla o fluxo é o código.

A marca do WORKFLOW: a ordem é decidida UMA VEZ, no início. A execução não
re-decide nada — se uma sala sujasse de novo no meio do caminho, o plano fixo
não perceberia. No Exemplo 04, o agente decide cada passo em tempo real e se adapta.

Dica: comece sempre pelo fluxo mais simples. Se um WORKFLOW resolve, não adicione
a complexidade (e o custo/latência) de um agente autônomo.

Execução
--------
- COM `ANTHROPIC_API_KEY`: o Claude gera o plano de limpeza.
- SEM chave: um planejador SIMULADO (mock) gera um plano de varredura equivalente.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 1) O AMBIENTE (idêntico aos Exemplos 01 e 02: apenas o ESTADO do mundo)
# ---------------------------------------------------------------------------
@dataclass
class Ambiente:
    """Estado do mundo: só dados."""

    salas: dict[str, bool]
    local: str
    desempenho: int = 0


# ---------------------------------------------------------------------------
# 2) PLANEJAMENTO (a etapa que usa o LLM — real ou simulado)
# ---------------------------------------------------------------------------
SYSTEM_PLANEJADOR = """Você é o planejador de um aspirador num corredor de quatro
salas em linha: A, B, C, D (A na ponta esquerda, D na ponta direita). Você recebe
a posição atual do aspirador e quais salas estão sujas, e deve devolver um PLANO:
a ordem em que as salas devem ser limpas.

Responda APENAS com a lista de salas separadas por vírgula (ex.: B, C, D, A).
Faça uma varredura eficiente, evitando idas e voltas desnecessárias."""


def planejar_com_llm(llm, ambiente: Ambiente) -> list[str]:
    """Pede ao Claude um plano (ordem das salas a limpar)."""
    sujas = [s for s, sujo in ambiente.salas.items() if sujo]
    pergunta = (
        f"Posição atual: {ambiente.local}. Salas sujas: {', '.join(sujas)}. "
        f"Qual o plano de limpeza?"
    )
    resposta = llm.invoke([
        {"role": "system", "content": SYSTEM_PLANEJADOR},
        {"role": "user", "content": pergunta},
    ])
    # Extrai apenas nomes de sala válidos da resposta do modelo.
    bruto = resposta.content.upper().replace("\n", ",")
    return [t.strip() for t in bruto.split(",") if t.strip() in ambiente.salas]


def planejar_mock(ambiente: Ambiente) -> list[str]:
    """Planejador SIMULADO: varre da posição atual até a parede e depois volta.

    A partir de B com tudo sujo, gera o plano B, C, D, A — exatamente o tipo de
    rota que o LLM produziria.
    """
    ordem = list(ambiente.salas.keys())
    idx = ordem.index(ambiente.local)

    plano: list[str] = []
    if ambiente.salas[ambiente.local]:           # a própria sala atual, se suja
        plano.append(ambiente.local)
    for s in ordem[idx + 1:]:                     # varre para a direita
        if ambiente.salas[s]:
            plano.append(s)
    for s in reversed(ordem[:idx]):               # depois volta para a esquerda
        if ambiente.salas[s]:
            plano.append(s)
    return plano


# ---------------------------------------------------------------------------
# 3) APRESENTAÇÃO (mesma visualização dos Exemplos 01 e 02)
# ---------------------------------------------------------------------------
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
# 4) EXECUÇÃO DETERMINÍSTICA DO PLANO (o controle do fluxo é do CÓDIGO)
# ---------------------------------------------------------------------------
def rodar_workflow(passos: int = 30) -> None:
    llm = construir_llm()
    usando_real = llm is not None

    ambiente = Ambiente(
        salas={"A": True, "B": True, "C": True, "D": True},
        local="B",
    )

    # --- ETAPA 1: PLANEJAR (uma única chamada de LLM, no início) ---
    plano = planejar_com_llm(llm, ambiente) if usando_real else planejar_mock(ambiente)

    rotulo = "LLM REAL (Claude)" if usando_real else "PLANEJADOR SIMULADO (mock)"
    historico: list[str] = [f"0. Plano definido ANTES da execução: {' → '.join(plano)}"]

    def desenhar(titulo: str) -> None:
        _limpar_tela()
        print("=" * 52)
        print("WORKFLOW (PLANEJAR → EXECUTAR) — Aspirador em 4 salas")
        print("=" * 52)
        print(f"Planejador: {rotulo}")
        print(f"Plano fixo: {' → '.join(plano)}\n")
        print(render(ambiente))
        print("\nLegenda:  V = aspirador   |   ::: = sujeira")
        print(f"Salas limpas: {salas_limpas(ambiente)}/{len(ambiente.salas)}\n")
        print(titulo)
        print("Histórico de execução:")
        for linha in historico:
            print("  " + linha)

    desenhar("Estado inicial (plano já decidido):")

    # --- ETAPA 2: EXECUTAR o plano na ordem (sem re-decidir nada) ---
    ordem = list(ambiente.salas.keys())
    passo = 0
    for alvo in plano:
        # Move passo a passo até a sala-alvo do plano.
        while ambiente.local != alvo and passo < passos:
            passo += 1
            i, j = ordem.index(ambiente.local), ordem.index(alvo)
            if j > i:
                ambiente.local = ordem[i + 1]
                historico.append(f"{passo}. Indo para {alvo}: move Direita → (agora em {ambiente.local})")
            else:
                ambiente.local = ordem[i - 1]
                historico.append(f"{passo}. Indo para {alvo}: move Esquerda ← (agora em {ambiente.local})")
            desenhar(f"Passo {passo}:")

        # Chegou no alvo: aspira.
        passo += 1
        if ambiente.salas[ambiente.local]:
            ambiente.salas[ambiente.local] = False
            ambiente.desempenho += 1
        historico.append(f"{passo}. Cheguei em {alvo}: Aspirar ↧")
        desenhar(f"Passo {passo}:")

    print(f"\nDesempenho final (PEAS - Performance): {ambiente.desempenho} salas limpas.")
    print(
        "\nRepare: o PLANO foi decidido UMA VEZ, no início (uma chamada de LLM), e o\n"
        "CÓDIGO apenas executou a ordem fixa. Isso é um WORKFLOW. No Exemplo 04, não há\n"
        "plano fixo: o próprio agente decide cada ação em tempo real — isso é um AGENTE."
    )


# ---------------------------------------------------------------------------
# Camada de LLM com fallback para mock (para rodar sem chave de API)
# ---------------------------------------------------------------------------
def construir_llm():
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None

    modelo = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    return ChatAnthropic(model=modelo, temperature=0)


if __name__ == "__main__":
    rodar_workflow()
