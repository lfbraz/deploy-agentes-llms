"""
base_vetorial.py — a base de conhecimento do agente (RAG) com embeddings do Databricks
======================================================================================

Retomamos a Aula 4 (bancos vetoriais e RAG). O agente não "sabe" a política da loja: ele a
RECUPERA de uma base de conhecimento indexada por embeddings e responde fundamentado nela.

Aqui usamos **embeddings reais do Databricks** (Foundation Model API) via `DatabricksEmbeddings`
— não há modo offline. O índice é mantido em memória com `numpy` (produto interno = similaridade
do cosseno); em produção, esta camada corresponde ao **Databricks Vector Search** (com
`DatabricksVectorSearchIndex` como recurso do agente). Requer `DATABRICKS_HOST`/`DATABRICKS_TOKEN`
e a lib `databricks-langchain`.
"""

from __future__ import annotations

import os

import numpy as np

from .mlflow_databricks import exigir_databricks

# Base de conhecimento da loja (chunks). Em produção viria de uma tabela/Delta indexada.
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


def _embed(textos: list[str]) -> np.ndarray:
    """Embeddings normalizados via endpoint do Databricks (Foundation Model API)."""
    exigir_databricks(libs=("databricks_langchain",))
    from databricks_langchain import DatabricksEmbeddings

    endpoint = os.getenv("ENDPOINT_EMBEDDINGS", "databricks-gte-large-en")
    cliente = DatabricksEmbeddings(endpoint=endpoint)
    vetores = np.asarray(cliente.embed_documents(textos), dtype="float32")
    normas = np.linalg.norm(vetores, axis=1, keepdims=True)
    return vetores / np.clip(normas, 1e-8, None)


class BaseVetorial:
    """Índice da base de conhecimento com busca por similaridade (top-k)."""

    def __init__(self, documentos: list[str] | None = None):
        self.documentos = documentos or POLITICAS
        self.matriz = _embed(self.documentos)

    def recuperar(self, pergunta: str, k: int = 2) -> list[str]:
        """Top-k trechos mais similares à pergunta (RECUPERAR do RAG)."""
        consulta = _embed([pergunta])[0]
        scores = self.matriz @ consulta
        indices = np.argsort(-scores)[:k]
        return [self.documentos[int(i)] for i in indices]


# Índice único reaproveitado pelos exemplos (construído sob demanda, na 1ª chamada).
_BASE: BaseVetorial | None = None


def recuperar(pergunta: str, k: int = 2) -> list[str]:
    """Atalho: recupera os k trechos mais relevantes da base de conhecimento."""
    global _BASE
    if _BASE is None:
        _BASE = BaseVetorial()
    return _BASE.recuperar(pergunta, k)


def backend_embeddings() -> str:
    """Nome do endpoint de embeddings em uso (para o log/observabilidade)."""
    return os.getenv("ENDPOINT_EMBEDDINGS", "databricks-gte-large-en")
