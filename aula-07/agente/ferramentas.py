"""
ferramentas.py — tool use com **Unity Catalog Functions**
=========================================================

Voltamos às Aulas 3-4 (loop de decisão e *tool use*): o agente não tenta responder tudo "de
cabeça"; quando falta um dado, ele decide ir buscar com uma ferramenta. No Databricks Agent
Framework, o jeito governado de oferecer uma ferramenta é registrá-la como **UC Function** no
Unity Catalog — vira um artefato com permissão, linhagem e busca, que qualquer agente pode
reaproveitar. Aqui temos duas:

  • `consultar_status_pedido(numero)` — pergunta ao "sistema de pedidos" (que aqui é simulado);
  • `calcular_frete(cep, peso)`        — faz a conta do frete por região e peso.

Repare que as funções são **autossuficientes** (carregam type hints, docstring e os imports no
próprio corpo): é assim que o `DatabricksFunctionClient.create_python_function` consegue
empacotá-las para o Unity Catalog. A busca na base de conhecimento (o RAG) fica de fora dessa
turma — ela não é uma UC Function, e sim a `VectorSearchRetrieverTool` (veja `base_vetorial.py`).
No fim, o agente recebe as duas coisas juntas: as UC Functions mais o retriever (veja
`agente_suporte.py`).
"""

from __future__ import annotations

from .config import FUNC_CALCULAR_FRETE, FUNC_STATUS_PEDIDO, SCHEMA, CATALOGO
from .mlflow_databricks import exigir_databricks


# ---------------------------------------------------------------------------------------
# As funções de negócio — cada uma se basta sozinha para virar uma UC Function.
# ---------------------------------------------------------------------------------------

def consultar_status_pedido(numero: str) -> str:
    """Consulta o status de um pedido pelo número no sistema de pedidos da loja.

    Args:
        numero: número do pedido a consultar, por exemplo '1002'.
    """
    import re

    pedidos = {
        "1001": "em separação no centro de distribuição",
        "1002": "a caminho — previsão de entrega em 2 dias úteis",
        "1003": "entregue em 12/06",
    }
    n = re.sub(r"\D", "", str(numero))
    if n in pedidos:
        return f"Pedido {n}: {pedidos[n]}."
    return f"Pedido {n} não encontrado. Confira o número na confirmação de compra."


def calcular_frete(cep: str, peso: float = 1.0) -> str:
    """Calcula o valor do frete a partir do CEP de destino e do peso em kg.

    Args:
        cep: CEP de destino (8 dígitos), por exemplo '01000-000'.
        peso: peso do pacote em quilogramas. Padrão 1.0.
    """
    import re

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


# ---------------------------------------------------------------------------------------
# O cliente que cria e executa as UC Functions no Unity Catalog.
# ---------------------------------------------------------------------------------------

_CLIENTE = None


def cliente_uc():
    """Monta o `DatabricksFunctionClient` uma vez só e o deixa como cliente padrão das tools."""
    global _CLIENTE
    if _CLIENTE is not None:
        return _CLIENTE
    exigir_databricks(libs=("unitycatalog",))
    from unitycatalog.ai.core.base import set_uc_function_client
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient

    _CLIENTE = DatabricksFunctionClient()
    set_uc_function_client(_CLIENTE)   # é este cliente que o UCFunctionToolkit vai usar
    return _CLIENTE


def registrar_uc_functions() -> list[str]:
    """Sobe as funções para o Unity Catalog como UC Functions (e devolve os nomes criados).

    Pode rodar à vontade: o `replace=True` simplesmente regrava a definição.
    """
    client = cliente_uc()
    for func in (consultar_status_pedido, calcular_frete):
        client.create_python_function(func=func, catalog=CATALOGO, schema=SCHEMA, replace=True)
    return [FUNC_STATUS_PEDIDO, FUNC_CALCULAR_FRETE]


def tools_uc():
    """Transforma as UC Functions em tools de agente, via `UCFunctionToolkit`.

    O toolkit puxa a assinatura e a descrição direto do Unity Catalog — e a mesma tool já sai
    pronta para LangChain, LlamaIndex ou os SDKs da OpenAI/Anthropic, sem retrabalho.
    """
    cliente_uc()
    from databricks_langchain import UCFunctionToolkit

    toolkit = UCFunctionToolkit(function_names=[FUNC_STATUS_PEDIDO, FUNC_CALCULAR_FRETE])
    return toolkit.tools
