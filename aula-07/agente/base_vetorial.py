"""
base_vetorial.py — RAG com **Databricks AI Search (Mosaic AI Vector Search)**
=============================================================================

Voltamos à Aula 4 (bancos vetoriais e RAG). O agente não decora a política da loja: ele vai
buscá-la numa base de conhecimento e responde com base no que encontrou. A diferença para a Aula 4
é o lar dessa base — agora ela é um **índice de Vector Search**, governado pelo Unity Catalog, e
não mais um índice solto na memória:

    tabela Delta (políticas)  ──►  índice delta-sync (Vector Search)  ──►  VectorSearchRetrieverTool

O melhor é que os **embeddings são por conta do Databricks**: a gente só aponta a coluna de texto
e o endpoint de embeddings, e o Vector Search se vira para calcular os vetores e mantê-los em dia
com a tabela-fonte (HNSW + busca híbrida via Reciprocal Rank Fusion). Para o agente, essa busca
chega como mais uma *tool* (`VectorSearchRetrieverTool`), do mesmo jeitinho que as UC Functions.

As funções que criam tudo isso (`criar_tabela_politicas`, `criar_indice`) são chamadas no
**setup** (exemplo 00). No dia a dia, o agente só toca no `retriever_tool()`.
"""

from __future__ import annotations

from .config import (
    ENDPOINT_EMBEDDINGS,
    ENDPOINT_VECTOR_SEARCH,
    INDICE_POLITICAS,
    TABELA_POLITICAS,
)
from .mlflow_databricks import exigir_databricks

# A base de conhecimento da loja, em pedacinhos. Na vida real, ela já viria de uma tabela Delta.
POLITICAS = [
    "Política de troca: você pode trocar ou devolver um produto em até 30 dias após o "
    "recebimento, apresentando a nota fiscal.",
    "Produtos em promoção não são elegíveis para troca; apenas para reembolso do valor pago.",
    "Frete: é grátis para compras acima de R$ 250 na região Sudeste; nas demais regiões é "
    "calculado por CEP e peso no checkout.",
    "Garantia: eletrônicos têm garantia padrão de 12 meses contra defeitos de fabricação.",
    "Atendimento humano: funciona de segunda a sexta, das 8h às 18h, pelo chat ou telefone.",
    "Reembolso: após aprovada a devolução, o estorno ocorre em até 7 dias úteis no mesmo meio "
    "de pagamento.",
]


# ---------------------------------------------------------------------------------------
# O setup (exemplo 00): montar a tabela-fonte e o índice. Pode rodar quantas vezes quiser.
# ---------------------------------------------------------------------------------------

def criar_tabela_politicas(spark) -> str:
    """Grava (ou regrava) a tabela Delta com as políticas — é dela que o índice se alimenta.

    O delta-sync só funciona com o **Change Data Feed** ligado na tabela, então já cuidamos disso
    aqui. Precisa da sessão `spark` (que o runtime do Databricks entrega de bandeja) e devolve o
    nome de três níveis da tabela.
    """
    linhas = [(i + 1, texto) for i, texto in enumerate(POLITICAS)]
    df = spark.createDataFrame(linhas, ["id", "texto"])
    (df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABELA_POLITICAS))
    spark.sql(
        f"ALTER TABLE {TABELA_POLITICAS} "
        "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )
    return TABELA_POLITICAS


def criar_indice() -> str:
    """Garante que o endpoint e o índice de Vector Search existem — criando-os só se faltarem."""
    exigir_databricks(libs=("databricks.vector_search",))
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient(disable_notice=True)

    # 1) O endpoint (o compute que serve o índice). Se já estiver de pé, seguimos em frente.
    nomes = {e["name"] for e in vsc.list_endpoints().get("endpoints", [])}
    if ENDPOINT_VECTOR_SEARCH not in nomes:
        vsc.create_endpoint_and_wait(name=ENDPOINT_VECTOR_SEARCH, endpoint_type="STANDARD")

    # 2) O índice em si, com os embeddings por conta do Databricks. Se já existe, não mexemos.
    try:
        vsc.get_index(ENDPOINT_VECTOR_SEARCH, INDICE_POLITICAS).describe()
    except Exception:
        vsc.create_delta_sync_index_and_wait(
            endpoint_name=ENDPOINT_VECTOR_SEARCH,
            index_name=INDICE_POLITICAS,
            source_table_name=TABELA_POLITICAS,
            pipeline_type="TRIGGERED",            # ressincroniza quando a gente pedir
            primary_key="id",
            embedding_source_column="texto",       # o Databricks calcula e mantém os vetores
            embedding_model_endpoint_name=ENDPOINT_EMBEDDINGS,
        )
    return INDICE_POLITICAS


# ---------------------------------------------------------------------------------------
# O que o agente realmente usa: a recuperação, embrulhada numa tool.
# ---------------------------------------------------------------------------------------

def retriever_tool():
    """A tool que faz a busca no índice de Vector Search — o "R" do RAG, em forma de ferramenta."""
    exigir_databricks(libs=("databricks_langchain",))
    from databricks_langchain import VectorSearchRetrieverTool

    # Índice com embeddings gerenciados: a coluna de texto já está declarada no próprio índice,
    # então NÃO passamos `text_column` aqui (o Vector Search reclama se a gente insistir).
    return VectorSearchRetrieverTool(
        index_name=INDICE_POLITICAS,
        num_results=3,
        columns=["id", "texto"],
        tool_name="buscar_politica",
        tool_description=(
            "Busca a política/FAQ da loja na base de conhecimento (troca, frete, garantia, "
            "reembolso, atendimento). Use sempre que a pergunta for sobre regras da loja."
        ),
    )


def backend_rag() -> str:
    """Só devolve o nome do índice em uso — útil para deixar registrado no log."""
    return INDICE_POLITICAS
