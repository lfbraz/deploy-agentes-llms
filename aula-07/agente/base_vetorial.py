"""
base_vetorial.py — RAG com **Databricks AI Search (Mosaic AI Vector Search)**
=============================================================================

Voltamos à Aula 4 (bancos vetoriais e RAG). O agente não decora a política da loja: ele a
busca numa base de conhecimento e responde com base no que encontrou. A diferença para a Aula 4
é onde essa base vive — agora ela é um **índice de Vector Search**, governado pelo Unity Catalog,
e não mais um índice solto na memória:

    tabela Delta (políticas)  ──►  índice delta-sync (Vector Search)  ──►  VectorSearchRetrieverTool

Os **embeddings ficam por conta do Databricks**: apontamos a coluna de texto e o endpoint de
embeddings, e o Vector Search calcula os vetores e os mantém sincronizados com a tabela-fonte
(HNSW + busca híbrida via Reciprocal Rank Fusion). Para o agente, essa busca chega como mais
uma *tool* (`VectorSearchRetrieverTool`), da mesma forma que as UC Functions.

As funções que criam tudo isso (`criar_tabela_politicas`, `criar_indice`) são chamadas no
**setup** (exemplo 00). Em tempo de execução, o agente usa apenas o `retriever_tool()`.
"""

from __future__ import annotations

from .config import (
    ENDPOINT_EMBEDDINGS,
    ENDPOINT_VECTOR_SEARCH,
    INDICE_POLITICAS,
    TABELA_POLITICAS,
)
from .mlflow_databricks import exigir_databricks

# A base de conhecimento da loja, em chunks. Em produção, ela viria de uma tabela Delta existente.
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

    O delta-sync exige o **Change Data Feed** ligado na tabela, então já cuidamos disso
    aqui. Precisa da sessão `spark` (disponível no runtime do Databricks) e devolve o
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


def criar_indice(timeout_min: int = 20) -> str:
    """Garante que o endpoint e o índice de Vector Search existem e estão prontos (ONLINE)."""
    import datetime

    exigir_databricks(libs=("databricks.vector_search",))
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient(disable_notice=True)
    timeout = datetime.timedelta(minutes=timeout_min)

    # 1) O endpoint (o compute que serve o índice). Se já estiver de pé, seguimos em frente.
    nomes = {e["name"] for e in vsc.list_endpoints().get("endpoints", [])}
    if ENDPOINT_VECTOR_SEARCH not in nomes:
        vsc.create_endpoint_and_wait(name=ENDPOINT_VECTOR_SEARCH, endpoint_type="STANDARD")

    # 2) O índice. Só criamos se ele NÃO existir (404); qualquer outro erro (ex.: 500) sobe.
    try:
        indice = vsc.get_index(ENDPOINT_VECTOR_SEARCH, INDICE_POLITICAS)
        estado = indice.describe()["status"]
    except Exception as e:
        if getattr(e, "status_code", None) != 404 and "NOT_FOUND" not in str(e).upper():
            raise
        indice = vsc.create_delta_sync_index(
            endpoint_name=ENDPOINT_VECTOR_SEARCH,
            index_name=INDICE_POLITICAS,
            source_table_name=TABELA_POLITICAS,
            pipeline_type="TRIGGERED",            # ressincroniza sob demanda
            primary_key="id",
            embedding_source_column="texto",       # o Databricks calcula e mantém os vetores
            embedding_model_endpoint_name=ENDPOINT_EMBEDDINGS,
        )
        estado = {"ready": False}
    else:
        # Já existe mas não está pronto (ex.: sync anterior falhou): dispara um novo sync.
        if not estado.get("ready"):
            print(f"Índice existe mas não está pronto ({estado.get('detailed_state')}); "
                  "disparando sync...")
            try:
                indice.sync()
            except Exception as e:
                if getattr(e, "status_code", None) != 409:   # 409 = já há um sync em andamento
                    raise

    # 3) Espera ficar ONLINE. No Free Edition, a falha típica aqui é a cota de serverless.
    if not estado.get("ready"):
        try:
            indice.wait_until_ready(verbose=True, timeout=timeout)
        except Exception as e:
            raise RuntimeError(
                f"O índice {INDICE_POLITICAS} não ficou pronto: {e}\n"
                "No Free Edition, o pipeline de sync costuma falhar por RESOURCE_EXHAUSTED (cota de "
                "compute serverless). Pare o SQL warehouse e outros computes serverless e rode de novo."
            ) from e
    return INDICE_POLITICAS


# ---------------------------------------------------------------------------------------
# O que o agente realmente usa: a recuperação, embrulhada numa tool.
# ---------------------------------------------------------------------------------------

def retriever_tool():
    """A tool que faz a busca no índice de Vector Search — o "R" do RAG, em forma de ferramenta."""
    exigir_databricks(libs=("databricks_langchain",))
    from databricks_langchain import VectorSearchRetrieverTool

    # Índice com embeddings gerenciados: a coluna de texto já está declarada no próprio índice,
    # então NÃO passamos `text_column` aqui (a validação rejeita o parâmetro nesse caso).
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
