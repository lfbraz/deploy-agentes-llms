"""
Exemplo 06 — O pipeline LLMOps de ponta a ponta (a jornada completa, ao vivo)
=============================================================================

Este exemplo amarra a aula inteira. LLMOps não é uma ferramenta, é um CICLO DE VIDA
operacional que leva o agente da bancada até produção com governança. Encadeamos os estágios
— exatamente o que o Databricks (MLflow + Unity Catalog) orquestra:

  DESENVOLVER  ->  REGISTRAR  ->  GATE DE CI/CD  ->  PROMOVER  ->  SERVIR  ->  MONITORAR
   (agente/)   (Prompt Registry)  (regressão)       (alias)    (resolve)    (traces)

Cada transição é registrada numa TRILHA DE AUDITORIA. Reaproveitamos os módulos reais dos
exemplos 02 (Prompt Registry no Unity Catalog) e 03 (gate sobre o agente real), provando que
as peças COMPÕEM. Requer DATABRICKS_HOST/TOKEN e `databricks-langchain`/`langgraph`.
"""

from __future__ import annotations

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agente.agente_suporte import responder            # noqa: E402
from agente.config import CONFIG_PADRAO                 # noqa: E402
from agente.mlflow_databricks import credenciais_ok      # noqa: E402


def _carregar(modulo_arquivo: str):
    """Importa um módulo vizinho cujo nome começa com dígito (import normal não permite)."""
    caminho = os.path.join(os.path.dirname(__file__), modulo_arquivo)
    nome = modulo_arquivo[:-3]
    spec = importlib.util.spec_from_file_location(nome, caminho)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    return mod


reg_mod = _carregar("02_registry_versionamento.py")     # Prompt Registry real
gate_mod = _carregar("03_gate_regressao_cicd.py")        # gate sobre o agente real


class PipelineLLMOps:
    """Orquestra o ciclo de vida e mantém a trilha de auditoria (governança)."""

    def __init__(self):
        self.auditoria: list[str] = []

    def _audit(self, etapa: str, detalhe: str) -> None:
        self.auditoria.append(f"[{etapa:10}] {detalhe}")

    def desenvolver(self) -> None:
        r = responder("Qual a garantia dos eletrônicos?")
        self._audit("DEV", f"agente respondeu via {', '.join(r.ferramentas_usadas) or 'LLM'} "
                           f"(trace {r.trace_id})")

    def registrar(self, template: str, autor: str):
        v = reg_mod.registrar_versao(template, CONFIG_PADRAO, autor)
        self._audit("REGISTRO", f"prompt v{v.version} por {autor}")
        return v

    def gate_e_promover(self, versao: int) -> bool:
        aval = gate_mod.avaliar_versao(CONFIG_PADRAO)
        aprovado = aval["taxa"] >= 0.75
        self._audit("CI/CD", f"v{versao}: taxa={aval['taxa']:.0%} -> "
                            f"{'APROVADO' if aprovado else 'REPROVADO'}")
        if aprovado:
            reg_mod.set_alias("champion", versao)
            self._audit("PROMOÇÃO", f"champion -> v{versao}")
        return aprovado

    def servir(self) -> str:
        champ = reg_mod.resolver("champion")
        self._audit("SERVING", f"tráfego roteado para champion = v{champ.version}")
        return f"v{champ.version}"

    def monitorar(self, perguntas: list[str]) -> None:
        for q in perguntas:
            responder(q)
        self._audit("MONITOR", f"{len(perguntas)} requisições rastreadas (traces/feedback)")

    def relatorio(self) -> str:
        return "\n".join(self.auditoria)


def main() -> int:
    print("=" * 76)
    print("PIPELINE LLMOps DE PONTA A PONTA — desenvolver→registrar→gate→promover→servir→monitorar")
    print("=" * 76)

    if not credenciais_ok():
        print("\n⚠️  O pipeline orquestra serviços REAIS do Databricks. Defina DATABRICKS_HOST/")
        print("    TOKEN e instale databricks-langchain/langgraph antes de rodar.")
        return 1
    print()

    pipe = PipelineLLMOps()
    pipe.desenvolver()                                                  # DESENVOLVER
    v = pipe.registrar(CONFIG_PADRAO.system_prompt, "ana")             # REGISTRAR
    pipe.gate_e_promover(v.version)                                     # GATE + PROMOVER
    servido = pipe.servir()                                             # SERVIR
    pipe.monitorar(["Qual o status do pedido 1002?", "Posso trocar em 20 dias?"])   # MONITORAR

    print(">> Trilha de auditoria (governança ponta a ponta):\n")
    print(pipe.relatorio())
    print(f"\n>> Produção está servindo: champion = {servido}")

    print(
        "\nLição: LLMOps é o ciclo — desenvolver, registrar, testar regressão, promover por\n"
        "alias, servir e monitorar — com cada passo AUDITADO. As peças das aulas anteriores\n"
        "(ferramentas, RAG, guardrails, avaliação, observabilidade) viram estágios governados\n"
        "dessa única esteira, e o monitoramento realimenta o gate a cada incidente."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
