"""
Passo 4 — Gate de aprovação: testes unitários do comportamento do agente
========================================================================

Nenhuma mudança no agente vai direto para o deploy: antes ela passa pelo gate — um conjunto de
**testes unitários** (pytest) que verificam o comportamento em **situações conhecidas**: as
políticas da base retornam a informação certa? O status do pedido vem dos **dados reais** da
tabela de pedidos no Unity Catalog?

É exatamente este arquivo que a esteira de CI/CD executa. Se um teste falhar, o `pytest`
sai com *exit code* != 0 e o **merge é bloqueado** (veja `ci/ci-cd-agente.yml`). A demo
`98_demo_gate_cicd.py` mostra isso na prática: remove o pedido 1002 da tabela para o gate
falhar e o recoloca para o gate voltar a passar.

    pytest aula-07/src/04_gate_aprovacao.py -v

Os testes executam o agente real (Claude + UC Functions + Vector Search), então precisam de
credenciais do Databricks. Sem elas, os testes são **pulados** (a esteira fica verde até você
configurar os segredos). Numa evolução natural, esses asserts poderiam virar *scorers*/
LLM-as-a-Judge (Aula 6); aqui ficamos na verificação determinística, mais simples e didática.
"""

from __future__ import annotations

import os
import sys

import pytest

# Deixa o pacote `agente` ao alcance do import, não importa de que pasta o pytest seja chamado.
_dir = os.path.dirname(os.path.abspath(__file__))
while _dir != os.path.dirname(_dir):
    if os.path.isdir(os.path.join(_dir, "agente")):
        sys.path.insert(0, _dir)
        break
    _dir = os.path.dirname(_dir)

from agente.agente_suporte import responder
from agente.mlflow_databricks import credenciais_ok

# Sem credenciais não dá para executar o agente real, então o módulo inteiro é pulado.
pytestmark = pytest.mark.skipif(
    not credenciais_ok(),
    reason="defina DATABRICKS_HOST/TOKEN para rodar o gate contra o agente real",
)


# ---------------------------------------------------------------------------------------
# Situações conhecidas — a resposta precisa trazer a informação certa.
# ---------------------------------------------------------------------------------------

def test_garantia_eletronicos_cita_12_meses():
    """Política que está na base -> o RAG tem de trazer os 12 meses de garantia."""
    r = responder("Qual a garantia dos eletrônicos?")
    assert "12 meses" in r.texto.lower(), f"esperava '12 meses', veio: {r.texto!r}"


def test_prazo_de_troca_30_dias():
    """Outra política conhecida -> a resposta tem de citar os 30 dias de troca."""
    r = responder("Posso trocar um produto comprado há 20 dias?")
    assert "30 dias" in r.texto.lower(), f"esperava '30 dias', veio: {r.texto!r}"


def test_status_pedido_usa_uc_function():
    """Pedido 1002 existe na tabela `pedidos` -> a UC Function traz o status real ("a caminho").

    Este teste depende dos DADOS: se o registro 1002 sumir da tabela do Unity Catalog, a
    função devolve "não encontrado" e o gate fica vermelho (é a demo do 98_demo_gate_cicd.py).
    """
    r = responder("Qual o status do pedido 1002?")
    assert "caminho" in r.texto.lower(), f"esperava o status 'a caminho', veio: {r.texto!r}"


# ---------------------------------------------------------------------------------------
# Rodando direto (sem pytest): executa os testes e devolve o exit code para a esteira.
# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
