"""
ferramentas.py — tool use com **Unity Catalog Functions**
=========================================================

Voltamos às Aulas 3-4 (loop de decisão e *tool use*): o agente não tenta responder tudo "de
cabeça"; quando falta um dado, ele decide ir buscar com uma ferramenta. No Databricks Agent
Framework, o jeito governado de oferecer uma ferramenta é registrá-la como **UC Function** no
Unity Catalog — vira um artefato com permissão, linhagem e busca, que qualquer agente pode
reaproveitar. Aqui temos duas, criadas por caminhos diferentes de propósito:

  • `consultar_status_pedido(numero)` — **função SQL** que consulta dados reais na tabela de
    pedidos do Unity Catalog (`aula_07.default.pedidos`);
  • `calcular_frete(cep, peso)`       — **função Python** que calcula o frete por região e peso.

A função SQL é o caminho certo quando a resposta está numa tabela: funções **Python** de UC
rodam num sandbox isolado, sem acesso a Spark nem à rede — elas não conseguem ler tabelas.
Já a função Python precisa ser **autossuficiente** (type hints, docstring e imports no próprio
corpo): é assim que o `DatabricksFunctionClient.create_python_function` a empacota para o
Unity Catalog. A busca na base de conhecimento (o RAG) não entra aqui — ela não é uma
UC Function, e sim a `VectorSearchRetrieverTool` (veja `base_vetorial.py`). No fim, o agente
recebe tudo junto: as UC Functions mais o retriever (veja `agente_suporte.py`).
"""

from __future__ import annotations

from .config import CATALOGO, FUNC_CALCULAR_FRETE, FUNC_STATUS_PEDIDO, SCHEMA, TABELA_PEDIDOS
from .mlflow_databricks import exigir_databricks


# ---------------------------------------------------------------------------------------
# A tabela de pedidos — os dados reais que a função SQL de status consulta.
# ---------------------------------------------------------------------------------------

# Registros de exemplo. O pedido 1002 ("a caminho") é o caso verificado pelo gate de
# aprovação (04_gate_aprovacao.py); a demo 98_demo_gate_cicd.py remove/recoloca esse
# registro para mostrar o gate barrando (ou liberando) o pipeline de CI/CD.
PEDIDOS = [
    ("1001", "em separação no centro de distribuição"),
    ("1002", "a caminho — previsão de entrega em 2 dias úteis"),
    ("1003", "entregue em 12/06"),
]


def criar_tabela_pedidos(spark) -> str:
    """Grava (ou regrava) a tabela Delta de pedidos — a fonte da UC Function de status.

    Precisa da sessão `spark` (runtime do Databricks) e devolve o nome de três níveis.
    """
    df = spark.createDataFrame(PEDIDOS, ["numero", "status"])
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABELA_PEDIDOS)
    return TABELA_PEDIDOS


# ---------------------------------------------------------------------------------------
# As funções de negócio: a consulta de status (SQL) e o cálculo de frete (Python).
# ---------------------------------------------------------------------------------------

def _sql_consultar_status_pedido() -> str:
    """O CREATE FUNCTION da consulta de status — função SQL com acesso à tabela de pedidos."""
    return f"""
CREATE OR REPLACE FUNCTION {FUNC_STATUS_PEDIDO}(
    numero STRING COMMENT 'número do pedido a consultar, por exemplo 1002'
)
RETURNS STRING
READS SQL DATA
COMMENT 'Consulta o status de um pedido pelo número no sistema de pedidos da loja (tabela {TABELA_PEDIDOS}).'
RETURN COALESCE(
    (SELECT concat('Pedido ', p.numero, ': ', p.status, '.')
       FROM {TABELA_PEDIDOS} p
      WHERE p.numero = regexp_replace(consultar_status_pedido.numero, '[^0-9]', '')
      LIMIT 1),
    concat('Pedido ', numero, ' não encontrado. Confira o número na confirmação de compra.')
)
"""


def calcular_frete(cep: str, peso: float) -> str:
    """Calcula o valor do frete a partir do CEP de destino e do peso em kg.

    Args:
        cep: CEP de destino (8 dígitos), por exemplo '01000-000'.
        peso: peso do pacote em quilogramas, por exemplo 1.0. (Sem valor padrão: a UC Function
            do Unity Catalog não aceita parâmetros com default; quando não souber, passe 1.0.)
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
    """Sobe as duas funções para o Unity Catalog como UC Functions (e devolve os nomes criados).

    Pode rodar mais de uma vez: o CREATE OR REPLACE / `replace=True` regrava a definição.
    A função SQL referencia a tabela de pedidos — rode `criar_tabela_pedidos` antes dela.
    """
    client = cliente_uc()
    client.create_function(sql_function_body=_sql_consultar_status_pedido())
    client.create_python_function(func=calcular_frete, catalog=CATALOGO, schema=SCHEMA,
                                  replace=True)
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
