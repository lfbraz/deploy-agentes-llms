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
configurar os segredos). Três formas de checar a resposta, da mais simples à mais flexível:
texto exato, regex (aceita variações) e um *scorer* LLM-as-a-judge (Aula 6).
"""

from __future__ import annotations

import os
import re
import sys

import pytest
from mlflow.genai.scorers import Correctness

# Deixa o pacote `agente` ao alcance do import, não importa de que pasta o pytest seja chamado.
_dir = os.path.dirname(os.path.abspath(__file__))
while _dir != os.path.dirname(_dir):
    if os.path.isdir(os.path.join(_dir, "agente")):
        sys.path.insert(0, _dir)
        break
    _dir = os.path.dirname(_dir)

from agente.agente_suporte import responder
from agente.config import ENDPOINT_LLM
from agente.mlflow_databricks import credenciais_ok

# Modelo juiz do scorer (mesmo formato da Aula 6). Por padrão, o próprio LLM do agente.
JUIZ = "databricks:/" + os.getenv("ENDPOINT_JUIZ", ENDPOINT_LLM)

# Sem credenciais não dá para executar o agente real, então o módulo inteiro é pulado.
pytestmark = pytest.mark.skipif(
    not credenciais_ok(),
    reason="defina DATABRICKS_HOST/TOKEN para rodar o gate contra o agente real",
)


# ---------------------------------------------------------------------------------------
# Situações conhecidas — a resposta precisa trazer a informação certa.
# ---------------------------------------------------------------------------------------

def test_garantia_eletronicos_cita_12_meses():
    """Política que está na base -> o RAG tem de trazer os 12 meses de garantia.

    Regex em vez de texto exato: o LLM pode escrever "12 meses", "doze meses" ou "1 ano".
    """
    r = responder("Qual a garantia dos eletrônicos?")
    assert re.search(r"\b(12|doze) meses\b|\b(1|um) ano\b", r.texto.lower()), \
        f"esperava '12 meses' (ou equivalente), veio: {r.texto!r}"


def test_prazo_de_troca_30_dias():
    """Outra política conhecida -> a resposta tem de citar os 30 dias de troca."""
    r = responder("Posso trocar um produto comprado há 20 dias?")
    assert "30 dias" in r.texto.lower(), f"esperava '30 dias', veio: {r.texto!r}"


def test_reembolso_correto_segundo_llm_juiz():
    """LLM-as-a-judge: o scorer `Correctness` (Aula 6) decide se a resposta traz o fato esperado.

    O juiz avalia o *significado* ("7" ou "sete" dias úteis) e devolve a justificativa (rationale).
    """
    pergunta = "Em quanto tempo recebo o estorno depois que a devolução é aprovada?"
    r = responder(pergunta)
    fb = Correctness(model=JUIZ)(
        inputs={"pergunta": pergunta},
        outputs=r.texto,
        expectations={"expected_facts": ["O estorno ocorre em até 7 dias úteis."]},
    )
    assert fb.value == "yes", f"juiz reprovou: {fb.rationale} | resposta: {r.texto!r}"


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
