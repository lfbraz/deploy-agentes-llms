"""
Exemplo 04 — Agente autônomo com Ferramentas (ReAct) no mundo do aspirador
==========================================================================

Objetivo didático
------------------
Fechamos o ciclo da aula, no MESMO mundo do aspirador — mas agora SEM plano fixo.
O AGENTE decide sozinho, a cada passo, a melhor ação: o padrão ReAct
(Reasoning + Acting), de Yao et al. (2022).

O caminho até aqui:
- Exemplo 01: decisão por regras IF/ELSE fixas (agente reativo).
- Exemplo 02: o LLM vira o MOTOR de decisão (mesmo laço, "cérebro" novo).
- Exemplo 03: WORKFLOW — o LLM faz UM plano no início e o código executa a ordem fixa.
- Exemplo 04 (aqui): o AGENTE decide em tempo real, escolhendo QUAIS ferramentas
  usar e em que ordem — e se adapta ao que observa.

Ferramentas que o agente pode chamar:
- `escanear_salas`: sensor — diz quais salas estão sujas e onde o aspirador está.
- `mover_para`: vai até uma sala (A, B, C ou D).
- `aspirar`: limpa a sala atual.

Diferença para o Exemplo 03: lá a ordem era decidida UMA VEZ. Aqui o agente
escaneia, decide o próximo passo (ex.: ir à sala suja mais próxima), age, observa
o resultado e repete — até concluir. É o fluxo dinâmico.

Execução
--------
- COM `ANTHROPIC_API_KEY`: usa o LangChain `create_agent` com Claude (agente real).
- SEM chave: roda um laço ReAct SIMULADO, com as mesmas ferramentas.
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
# 1) O AMBIENTE (idêntico aos exemplos anteriores: apenas o ESTADO do mundo)
# ---------------------------------------------------------------------------
@dataclass
class Ambiente:
    """Estado do mundo: só dados."""

    salas: dict[str, bool]
    local: str
    desempenho: int = 0


# ---------------------------------------------------------------------------
# 2) APRESENTAÇÃO (mesma visualização dos exemplos anteriores)
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
# 3) FERRAMENTAS (Tools) — o agente age sobre o ambiente por meio delas
# ---------------------------------------------------------------------------
def construir_ferramentas(ambiente: Ambiente, historico: list[str], desenhar) -> dict:
    """Cria as ferramentas ligadas a ESTE ambiente.

    Cada ferramenta altera o estado do mundo, registra no histórico e redesenha
    a cena — assim a turma vê CADA ação que o agente decide tomar.
    """
    ordem = list(ambiente.salas.keys())

    def escanear_salas() -> str:
        """Informa quais salas estão sujas e em qual sala o aspirador está."""
        sujas = [s for s in ordem if ambiente.salas[s]]
        texto = ", ".join(sujas) if sujas else "nenhuma"
        historico.append(f"escanear_salas() → sujas: {texto} | posição: {ambiente.local}")
        desenhar("Ação: escanear_salas()")
        return f"Posição atual: {ambiente.local}. Salas sujas: {texto}."

    def mover_para(sala: str) -> str:
        """Move o aspirador para a sala indicada (use uma letra: A, B, C ou D)."""
        alvo = sala.strip().upper()[:1]
        if alvo not in ambiente.salas:
            return f"Sala inválida: {sala}. Use A, B, C ou D."
        ambiente.local = alvo
        historico.append(f"mover_para({alvo}) → aspirador agora em {alvo}")
        desenhar(f"Ação: mover_para({alvo})")
        return f"Aspirador agora na sala {alvo}."

    def aspirar() -> str:
        """Aspira (limpa) a sala onde o aspirador está agora."""
        if ambiente.salas[ambiente.local]:
            ambiente.salas[ambiente.local] = False
            ambiente.desempenho += 1
            historico.append(f"aspirar() → sala {ambiente.local} limpa ✓")
            desenhar("Ação: aspirar()")
            return f"Sala {ambiente.local} limpa."
        historico.append(f"aspirar() → sala {ambiente.local} já estava limpa")
        desenhar("Ação: aspirar()")
        return f"Sala {ambiente.local} já estava limpa."

    return {"escanear_salas": escanear_salas, "mover_para": mover_para, "aspirar": aspirar}


SYSTEM_PROMPT = (
    "Você é um aspirador de pó autônomo num corredor de quatro salas em linha: "
    "A, B, C, D (A na ponta esquerda, D na ponta direita). Seu objetivo é deixar "
    "TODAS as salas limpas, da forma mais eficiente possível. Use 'escanear_salas' "
    "para descobrir o que está sujo, 'mover_para' para se deslocar e 'aspirar' para "
    "limpar. Vá sempre à sala suja mais próxima da sua posição atual. Pare quando "
    "não houver mais salas sujas."
)


# ---------------------------------------------------------------------------
# 4) CAMINHO 1 — Agente REAL com LangChain create_agent
# ---------------------------------------------------------------------------
def rodar_agente_real(ambiente: Ambiente, ferramentas: dict) -> str:
    from langchain.agents import create_agent

    modelo = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    agente = create_agent(
        model=f"anthropic:{modelo}",
        tools=list(ferramentas.values()),
        system_prompt=SYSTEM_PROMPT,
    )
    resultado = agente.invoke(
        {"messages": [{"role": "user", "content": "Limpe todas as salas sujas do corredor."}]}
    )
    return resultado["messages"][-1].content


# ---------------------------------------------------------------------------
# 5) CAMINHO 2 — Laço ReAct SIMULADO (sem API), com as MESMAS ferramentas
# ---------------------------------------------------------------------------
def rodar_agente_mock(ambiente: Ambiente, ferramentas: dict, historico: list[str]) -> str:
    ordem = list(ambiente.salas.keys())

    while any(ambiente.salas.values()):
        # PENSAMENTO: olho o estado e escolho a sala suja mais próxima.
        ferramentas["escanear_salas"]()
        sujas = [s for s in ordem if ambiente.salas[s]]
        atual = ordem.index(ambiente.local)
        alvo = min(sujas, key=lambda s: abs(ordem.index(s) - atual))
        historico.append(f"Pensamento: a sala suja mais próxima de {ambiente.local} é {alvo}. Vou até lá.")

        # AÇÃO: movo (se preciso) e aspiro.
        if ambiente.local != alvo:
            ferramentas["mover_para"](alvo)
        ferramentas["aspirar"]()

    return "Terminei: todas as salas estão limpas."


# ---------------------------------------------------------------------------
# 6) Orquestração da demonstração
# ---------------------------------------------------------------------------
def main() -> None:
    ambiente = Ambiente(
        salas={"A": True, "B": True, "C": True, "D": True},
        local="B",
    )
    historico: list[str] = ["0. Início: aspirador na sala B, sujeira em todas."]

    usando_real = bool(os.getenv("ANTHROPIC_API_KEY"))
    rotulo = "AGENTE REAL (LangChain + Claude)" if usando_real else "AGENTE SIMULADO (mock)"

    def desenhar(titulo: str) -> None:
        _limpar_tela()
        print("=" * 54)
        print("AGENTE AUTÔNOMO (ReAct + ferramentas) — Aspirador")
        print("=" * 54)
        print(f"Modo: {rotulo}\n")
        print(render(ambiente))
        print("\nLegenda:  V = aspirador   |   ::: = sujeira")
        print(f"Salas limpas: {salas_limpas(ambiente)}/{len(ambiente.salas)}\n")
        print(titulo)
        print("Histórico (decisões e ferramentas):")
        for linha in historico:
            print("  " + linha)

    ferramentas = construir_ferramentas(ambiente, historico, desenhar)
    desenhar("Estado inicial:")

    resposta_final = None
    if usando_real:
        try:
            resposta_final = rodar_agente_real(ambiente, ferramentas)
        except ImportError:
            historico.append("(LangChain não instalado — caindo para o modo simulado)")
            resposta_final = rodar_agente_mock(ambiente, ferramentas, historico)
    else:
        resposta_final = rodar_agente_mock(ambiente, ferramentas, historico)

    desenhar("Concluído:")
    print(f"\nResposta final do agente:\n  {resposta_final}")
    print(f"\nDesempenho final (PEAS - Performance): {ambiente.desempenho} salas limpas.")
    print(
        "\nÉ o MESMO laço percepção -> decisão -> ação do aspirador, mas agora o agente\n"
        "escolhe AÇÕES (ferramentas) e a ordem delas em TEMPO REAL — sem plano fixo.\n"
        "Compare com o Exemplo 03: lá a rota era decidida antes; aqui o agente decide\n"
        "e se adapta a cada observação. Esse é o coração de um agente autônomo."
    )


if __name__ == "__main__":
    main()
