"""
guardrails.py — segurança do agente: filtros de ENTRADA e de SAÍDA
==================================================================

Retomamos a Aula 5 (segurança e guardrails). Um agente em produção é cercado por duas
camadas de proteção determinísticas, que rodam ANTES e DEPOIS do modelo:

  • Guardrail de ENTRADA: barra *prompt injection* / *jailbreak* (ex.: "ignore as
    instruções anteriores", "aja como DAN") e detecta PII que não deveria circular.
  • Guardrail de SAÍDA: impede que a resposta vaze PII ou segredos (ex.: o system prompt,
    chaves, dados de outros clientes).

Aqui implementamos versões **em código, determinísticas e offline** (sem chave de API),
suficientes para o didático e para servir de checagem reproduzível na esteira de CI/CD.
Em produção, estes filtros correspondem a serviços dedicados — **Databricks AI Gateway**
(PII, *unsafe content*, *jailbreak*), **Azure Prompt Shields** ou **Llama Guard** —, que a
Aula 5 cobriu. A interface (entra texto, sai um veredito) é a mesma.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Padrões de jailbreak / prompt injection mais comuns (proxy das listas in-the-wild da Aula 5).
_PADROES_INJECAO = [
    r"ignore (as |todas as )?(instru[cç][oõ]es|regras)",
    r"esque[çc]a (as |suas )?(instru[cç][oõ]es|regras)",
    r"aja como (dan|um modelo sem regras|se n[aã]o houvesse regras)",
    r"modo desenvolvedor|developer mode|do anything now",
    r"revele (o |seu )?(system ?prompt|prompt do sistema|instru[cç][oõ]es internas)",
    r"ignore (the )?(previous|above) (instructions|rules)",
]

# PII: CPF (000.000.000-00 ou 11 dígitos), e-mail, cartão (16 dígitos).
_CPF = re.compile(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2})\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_CARTAO = re.compile(r"\b(?:\d[ -]?){16}\b")
# Marcadores de segredo que jamais devem aparecer na SAÍDA.
_SEGREDOS = ["system prompt", "prompt do sistema", "sk-", "api_key", "senha=", "password="]


@dataclass
class Veredito:
    """Resultado de um guardrail: passou ou bloqueou, com o motivo e o texto tratado."""

    bloqueado: bool
    motivo: str = "ok"
    texto: str = ""                       # texto possivelmente higienizado (PII mascarada)
    sinais: list[str] = field(default_factory=list)


def _mascarar_pii(texto: str) -> tuple[str, list[str]]:
    """Substitui PII por marcadores e devolve (texto_mascarado, sinais encontrados)."""
    sinais: list[str] = []
    novo = texto
    if _CPF.search(novo):
        sinais.append("cpf"); novo = _CPF.sub("[CPF]", novo)
    if _CARTAO.search(novo):
        sinais.append("cartao"); novo = _CARTAO.sub("[CARTAO]", novo)
    if _EMAIL.search(novo):
        sinais.append("email"); novo = _EMAIL.sub("[EMAIL]", novo)
    return novo, sinais


def guardrail_entrada(texto: str) -> Veredito:
    """Filtra a pergunta do usuário ANTES de chegar ao modelo."""
    t = texto.lower()
    for padrao in _PADROES_INJECAO:
        if re.search(padrao, t):
            return Veredito(True, "tentativa de prompt injection/jailbreak detectada",
                            texto, ["jailbreak"])
    texto_seguro, sinais = _mascarar_pii(texto)
    if sinais:
        # PII não bloqueia, mas é mascarada antes de seguir (minimização de dados).
        return Veredito(False, f"PII mascarada na entrada: {', '.join(sinais)}",
                        texto_seguro, sinais)
    return Veredito(False, "ok", texto, [])


def guardrail_saida(texto: str) -> Veredito:
    """Filtra a resposta do modelo ANTES de devolver ao usuário."""
    t = texto.lower()
    for marcador in _SEGREDOS:
        if marcador in t:
            return Veredito(True, f"possível vazamento de segredo: '{marcador}'",
                            "Desculpe, não posso compartilhar essa informação.", ["vazamento"])
    texto_seguro, sinais = _mascarar_pii(texto)
    if sinais:
        return Veredito(False, f"PII mascarada na saída: {', '.join(sinais)}",
                        texto_seguro, sinais)
    return Veredito(False, "ok", texto, [])
