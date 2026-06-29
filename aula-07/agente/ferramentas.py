"""
ferramentas.py — tool use: as ferramentas que o agente pode chamar
==================================================================

Retomamos as Aulas 3 e 4 (loop de decisão e *tool use*). O agente não responde tudo "de
cabeça": quando precisa de um dado, ele DECIDE chamar uma ferramenta. Definimos três:

  • `consultar_status_pedido(numero)` — consulta um "sistema de pedidos" (simulado);
  • `calcular_frete(cep, peso)`        — calcula frete por região/peso (regra simulada);
  • `buscar_politica(pergunta)`        — recupera a política da loja na base vetorial (RAG).

Cada ferramenta é uma função Python pura; `ferramentas_langchain()` as embrulha como
ferramentas LangChain (`@tool`) e as entrega ao grafo do agente, que decide quando chamá-las.
O import do LangChain é lazy (acontece só ao construir o agente).
"""

from __future__ import annotations

import re

from .base_vetorial import recuperar

# "Banco de pedidos" simulado (em produção, uma chamada de API/SQL com timeout e retry).
_PEDIDOS = {
    "1001": "em separação no centro de distribuição",
    "1002": "a caminho — previsão de entrega em 2 dias úteis",
    "1003": "entregue em 12/06",
}


def consultar_status_pedido(numero: str) -> str:
    """Consulta o status de um pedido pelo número. Ex.: numero='1002'."""
    numero = re.sub(r"\D", "", str(numero))
    if numero in _PEDIDOS:
        return f"Pedido {numero}: {_PEDIDOS[numero]}."
    return f"Pedido {numero} não encontrado. Confira o número na confirmação de compra."


def calcular_frete(cep: str, peso: float = 1.0) -> str:
    """Calcula o frete por CEP e peso (kg). Ex.: cep='01000-000', peso=2.0."""
    digitos = re.sub(r"\D", "", str(cep))
    if len(digitos) < 8:
        return "CEP inválido — informe os 8 dígitos."
    try:
        peso = float(peso)
    except (TypeError, ValueError):
        peso = 1.0
    regiao = "Sudeste" if digitos[0] in "012" else "demais regiões"
    base = 12.0 if regiao == "Sudeste" else 22.0
    valor = base + 4.5 * max(0.0, peso - 1.0)
    return f"Frete para o CEP {digitos} ({regiao}, {peso:.1f} kg): R$ {valor:.2f}."


def buscar_politica(pergunta: str) -> str:
    """Busca na base de conhecimento da loja (políticas/FAQ) e devolve os trechos relevantes."""
    trechos = recuperar(pergunta, k=2)
    if not trechos:
        return "Nenhuma política relevante encontrada."
    return " ".join(trechos)


def ferramentas_langchain():
    """Embrulha as funções como ferramentas LangChain (`@tool`) para o agente.

    Import preguiçoso: só é chamado quando o agente Databricks é construído (libs presentes).
    """
    from langchain_core.tools import tool

    @tool
    def status_pedido(numero: str) -> str:
        """Consulta o status de um pedido pelo número (ex.: '1002')."""
        return consultar_status_pedido(numero)

    @tool
    def frete(cep: str, peso: float = 1.0) -> str:
        """Calcula o frete por CEP (8 dígitos) e peso em kg."""
        return calcular_frete(cep, peso)

    @tool
    def politica(pergunta: str) -> str:
        """Busca a política/FAQ da loja na base de conhecimento (troca, frete, garantia, reembolso)."""
        return buscar_politica(pergunta)

    return [status_pedido, frete, politica]
