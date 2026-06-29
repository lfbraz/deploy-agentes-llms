"""
Exemplo 02 — O LLM como "motor" do agente
==========================================

Objetivo didático
------------------
Este é o MESMO mundo do Exemplo 01: o aspirador reativo num corredor de quatro
salas (A, B, C, D). Reaproveitamos tudo — o `Ambiente` (só dados), os SENSORES
(perceber), os ATUADORES (agir) e a VISUALIZAÇÃO. A ÚNICA coisa que muda é o
`decidir`: no Exemplo 01 ele eram regras IF/ELSE fixas; aqui a decisão passa a
vir de um LLM raciocinando sobre a percepção descrita em linguagem natural.

Esse é o salto conceitual da aula: o LLM vira o "cérebro" (motor de decisão)
dentro do mesmo laço percepção -> decisão -> ação.

Execução
--------
- COM `ANTHROPIC_API_KEY` no ambiente: usa o Claude de verdade como motor.
- SEM chave: cai automaticamente num "LLM simulado" (mock) que reproduz a mesma
  estratégia, para você estudar o fluxo sem internet ou credenciais.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Carrega variáveis do .env, se existir (não falha se não houver python-dotenv).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 1) O AMBIENTE (idêntico ao Exemplo 01: apenas o ESTADO do mundo)
# ---------------------------------------------------------------------------
@dataclass
class Ambiente:
    """Estado do mundo."""

    salas: dict[str, bool]
    local: str
    desempenho: int = 0


# ---------------------------------------------------------------------------
# 2) O MOTOR DE DECISÃO (LLM real ou simulado)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Você é o motor de decisão de um aspirador de pó autônomo.
O mundo é um corredor com QUATRO salas em linha: A, B, C, D (A na ponta esquerda,
D na ponta direita). A cada passo você recebe a percepção: a sala atual, se ela
está suja, se há parede à esquerda/direita e o sentido em que vinha andando.

Responda com EXATAMENTE UMA palavra entre: Aspirar, Esquerda, Direita.

Estratégia: se a sala atual está suja, Aspirar. Caso contrário, continue varrendo
no sentido atual; ao bater numa parede, inverta o sentido. Responda só a ação."""


def construir_motor_llm():
    """Tenta construir um motor baseado no Claude. Se não der, retorna None."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None

    modelo = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    llm = ChatAnthropic(model=modelo, temperature=0)

    def motor(percepcao: dict, sentido: str) -> str:
        pergunta = (
            f"Percepção: sala={percepcao['local']}, suja={percepcao['sujo']}, "
            f"parede_esquerda={percepcao['parede_esquerda']}, "
            f"parede_direita={percepcao['parede_direita']}, "
            f"sentido_atual={sentido}. Qual a ação?"
        )
        resposta = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pergunta},
        ])
        return resposta.content.strip().split()[0]  # primeira palavra

    return motor


def motor_mock(percepcao: dict, sentido: str) -> str:
    """LLM SIMULADO: reproduz a MESMA estratégia do prompt, sem chamar a API.

    É exatamente a política do Exemplo 01 — serve para ver o agente funcionando
    mesmo sem chave de API.
    """
    if percepcao["sujo"]:
        return "Aspirar"
    if percepcao["parede_direita"]:
        return "Esquerda"
    if percepcao["parede_esquerda"]:
        return "Direita"
    return sentido


# ---------------------------------------------------------------------------
# 3) O AGENTE (sensores + atuadores iguais ao 01; o decidir agora usa o motor)
# ---------------------------------------------------------------------------
class AspiradorComLLM:
    """Mesmo agente do Exemplo 01, mas com o LLM no lugar das regras fixas."""

    def __init__(self, motor, sentido: str = "Direita"):
        self.motor = motor      # função (percepcao, sentido) -> ação
        self.sentido = sentido  # única "memória": o sentido da varredura

    # --- SENSORES (idêntico ao Exemplo 01) ---
    def perceber(self, ambiente: Ambiente) -> dict:
        ordem = list(ambiente.salas.keys())
        idx = ordem.index(ambiente.local)
        return {
            "local": ambiente.local,
            "sujo": ambiente.salas[ambiente.local],
            "parede_esquerda": idx == 0,
            "parede_direita": idx == len(ordem) - 1,
        }

    # --- DECISÃO: agora delegada ao MOTOR (LLM real ou simulado) ---
    def decidir(self, percepcao: dict) -> str:
        acao = self.motor(percepcao, self.sentido)
        if acao in ("Esquerda", "Direita"):
            self.sentido = acao  # lembra o novo sentido da varredura
        return acao

    # --- ATUADORES (idêntico ao Exemplo 01) ---
    def agir(self, ambiente: Ambiente, acao: str) -> None:
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
# 4) APRESENTAÇÃO (idêntica ao Exemplo 01)
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
# 5) O LOOP PERCEPÇÃO-AÇÃO (igual ao 01, mas a decisão vem do LLM)
# ---------------------------------------------------------------------------
def rodar_simulacao(passos: int = 30) -> None:
    motor = construir_motor_llm()
    usando_real = motor is not None
    if motor is None:
        motor = motor_mock

    ambiente = Ambiente(
        salas={"A": True, "B": True, "C": True, "D": True},
        local="B",
    )
    aspirador = AspiradorComLLM(motor=motor, sentido="Direita")

    rotulo_motor = "LLM REAL (Claude)" if usando_real else "LLM SIMULADO (mock)"
    historico: list[str] = ["0. Início: aspirador na sala B, sujeira em todas."]

    def desenhar(titulo: str) -> None:
        _limpar_tela()
        print("=" * 46)
        print("LLM COMO MOTOR DE DECISÃO — Aspirador em 4 salas")
        print("=" * 46)
        print(f"Motor de decisão: {rotulo_motor}\n")
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
        acao = aspirador.decidir(percepcao)          # DECISÃO (agora é o LLM)
        aspirador.agir(ambiente, acao)               # AÇÃO (atuadores)

        estado = "suja" if percepcao["sujo"] else "limpa"
        historico.append(
            f"{passo}. Sala {percepcao['local']} ({estado}): LLM decidiu {acao}  {SETA[acao]}"
        )

        desenhar(f"Passo {passo}:")

        if tudo_limpo(ambiente):
            break

    print(f"\nDesempenho final (PEAS - Performance): {ambiente.desempenho} salas limpas.")
    print(
        "\nObserve: a MESMA estrutura do Exemplo 01 (Ambiente, sensores, atuadores e\n"
        "visualização). A ÚNICA mudança foi o `decidir`: trocamos regras IF/ELSE fixas\n"
        "por um LLM. No Exemplo 03 subimos um nível: orquestrar VÁRIAS chamadas de LLM."
    )


if __name__ == "__main__":
    rodar_simulacao()
