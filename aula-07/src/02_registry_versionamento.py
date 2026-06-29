"""
Exemplo 02 — Registrar e versionar o agente: MLflow Prompt Registry (Unity Catalog)
===================================================================================

Com o agente pronto (exemplo 01), o segundo passo da jornada é VERSIONAR. Diferente do MLOps
clássico (onde se versiona pesos de um modelo treinado), num agente LLM o "artefato" que mais
muda é o **prompt** (parte da tripla `ConfigAgente`). Usamos o **MLflow Prompt Registry** REAL,
no **Unity Catalog**, que dá versionamento imutável estilo Git + **aliases móveis**:

  - `register_prompt(name, template, ...)` cria uma VERSÃO imutável (v1, v2, v3...);
  - `set_prompt_alias(name, alias, version)` move um alias (`champion`, `challenger`);
  - `load_prompt("prompts:/<name>@<alias>")` é como o código de produção resolve a versão.

Promover = mover o alias `champion`; ROLLBACK = mover o alias de volta. A versão é imutável; o
alias é o ponteiro. Essa indireção torna deploy e rollback uma operação O(1).

Requer DATABRICKS_HOST/TOKEN (e o catálogo/schema do `MODELO_UC`, padrão `main.default`).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agente.config import CONFIG_PADRAO, ConfigAgente              # noqa: E402
from agente.mlflow_databricks import (                              # noqa: E402
    configurar_mlflow, credenciais_ok, nome_uc,
)

NOME_PROMPT = nome_uc("agente_suporte_prompt")     # ex.: main.default.agente_suporte_prompt


def registrar_versao(template: str, config: ConfigAgente, autor: str):
    """Registra uma nova versão imutável do prompt no Prompt Registry (Unity Catalog)."""
    configurar_mlflow()
    import mlflow

    return mlflow.genai.register_prompt(
        name=NOME_PROMPT,
        template=template,
        commit_message=f"versão por {autor}",
        tags={"autor": autor, "endpoint": config.endpoint,
              "temperatura": str(config.temperatura)},
    )


def set_alias(alias: str, versao: int) -> None:
    """Aponta um alias (champion/challenger/...) para uma versão registrada."""
    import mlflow
    mlflow.genai.set_prompt_alias(NOME_PROMPT, alias=alias, version=versao)


def resolver(alias: str):
    """Resolve 'prompts:/<name>@<alias>' -> a versão apontada (como em produção)."""
    import mlflow
    return mlflow.genai.load_prompt(f"prompts:/{NOME_PROMPT}@{alias}")


def main() -> int:
    print("=" * 72)
    print("REGISTRAR E VERSIONAR O AGENTE — MLflow Prompt Registry + aliases (Unity Catalog)")
    print("=" * 72)

    if not credenciais_ok():
        print("\n⚠️  O registry roda no Unity Catalog. Defina DATABRICKS_HOST/TOKEN (e, se preciso,")
        print("    MODELO_UC=catalogo.schema.objeto) antes de rodar.")
        return 1

    print(f"\nPrompt registrado em: {NOME_PROMPT}\n")

    # Ciclo de experimentação: três versões do system prompt (a tripla evoluindo).
    v1 = registrar_versao("Você é um atendente. Responda em 1 frase.", CONFIG_PADRAO, "luiz")
    v2 = registrar_versao("Você é um atendente cordial. Responda em 1 frase e cite a política.",
                          CONFIG_PADRAO, "luiz")
    v3 = registrar_versao("Você é um atendente cordial. Responda em 1 frase, cite a política e "
                          "ofereça o próximo passo.", CONFIG_PADRAO, "ana")
    print(f"   versões registradas: v{v1.version}, v{v2.version}, v{v3.version}")

    # Em produção, v2 é o champion; v3 entra como challenger (canário).
    set_alias("champion", v2.version)
    set_alias("challenger", v3.version)
    print("\n>> O que o código de produção resolve (via alias, não via número):")
    print(f"   prompts:/{NOME_PROMPT}@champion   -> v{resolver('champion').version}")
    print(f"   prompts:/{NOME_PROMPT}@challenger -> v{resolver('challenger').version}")

    # O challenger venceu os testes -> promover a champion (deploy O(1)).
    print("\n>> Challenger aprovado: promovendo a champion...")
    set_alias("champion", v3.version)
    print(f"   prompts:/{NOME_PROMPT}@champion   -> v{resolver('champion').version}")

    # Incidente em produção -> rollback imediato para a versão anterior.
    print("\n>> Incidente! Rollback do champion para a versão anterior...")
    set_alias("champion", v2.version)
    print(f"   prompts:/{NOME_PROMPT}@champion   -> v{resolver('champion').version}")

    print(
        "\nLição: versione o prompt como artefato IMUTÁVEL no Prompt Registry e deixe ALIASES\n"
        "móveis decidirem o que vai a produção. Deploy e rollback viram um reapontamento de\n"
        "ponteiro — a base de blue-green e champion/challenger."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
