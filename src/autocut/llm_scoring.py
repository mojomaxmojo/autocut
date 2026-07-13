"""LLM-Segment-Scoring (Schritt 7 aus FEATURE-PLAN.md, optionaler
KI-Layer - nur aktiv wenn --no-ai NICHT gesetzt ist UND ein API_KEY in
.env vorhanden ist).

Bewertet jedes Transkript-Segment (aus Schritt 6) per Cloud-Free-Tier-
LLM-API (Groq oder OpenRouter, beide OpenAI-kompatible Chat-Completions-
Schnittstellen) auf einer Skala 0-10, wie "highlight-wuerdig" es fuer
ein Reise-/Naturvideo ist.

Kernprinzip: Fehlt der API_KEY, oder schlaegt ein Request fehl, wird
IMMER None zurueckgegeben (fuer das gesamte Ergebnis oder pro Segment)
- niemals eine Exception. Die Pipeline muss auch ohne dieses Cloud-
Scoring vollstaendig funktionieren (siehe scoring.fuse_scores(), das
mit llm_scores=None bereits umgehen kann).
"""

from __future__ import annotations

import logging
import time

import requests

from .config import LlmConfig

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_SYSTEM_PROMPT = (
    "Du bewertest Transkript-Segmente aus einem deutschen Reise-/"
    "Wohnmobil-/Landschafts-Video. Antworte AUSSCHLIESSLICH mit einer "
    "einzelnen Ganzzahl von 0 bis 10, ohne jeden weiteren Text."
)


def build_prompt(segment_text: str) -> str:
    """Baut den Bewertungs-Prompt fuer ein einzelnes Transkript-Segment.

    Besonders hoch bewerten: Ankunft an einem Ort, Wetterbeschreibungen,
    Ausrufe/Begeisterung, Tierbeobachtungen, Aussichts-/Sonnenuntergangs-
    Erwaehnungen, wichtige Reise-Informationen. Niedrig bewerten:
    Fuellwoerter, Wiederholungen, belangloses Reden.
    """
    return (
        "Bewerte folgendes Transkript-Segment aus einem Reise-/Naturvideo "
        "auf einer Skala von 0 (uninteressant) bis 10 (absolutes Highlight), "
        "wie \"highlight-wuerdig\" es ist.\n\n"
        "Besonders HOCH bewerten (8-10): Ankunft an einem neuen Ort, "
        "Wetterbeschreibungen (z.B. Sturm, Regenbogen), Ausrufe/Begeisterung "
        "(z.B. \"wow\", \"schau mal\"), Tierbeobachtungen, Erwaehnungen von "
        "Aussicht/Sonnenuntergang, wichtige Informationen zur Reise.\n"
        "Niedrig bewerten (0-3): Fuellwoerter, Wiederholungen, belangloses "
        "Reden ohne besonderen Inhalt.\n\n"
        f"Segment: \"{segment_text}\"\n\n"
        "Antworte NUR mit einer Ganzzahl von 0 bis 10."
    )


def _parse_score_response(content: str) -> float | None:
    """Extrahiert eine Zahl 0-10 aus der LLM-Antwort. Robust gegenueber
    zusaetzlichem Text (z.B. "8" oder "Score: 8" oder "8/10")."""
    cleaned = content.strip()
    digits = ""
    for ch in cleaned:
        if ch.isdigit() or (ch == "." and digits):
            digits += ch
        elif digits:
            break
    if not digits:
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    return max(0.0, min(10.0, value))


def _call_chat_completions(
    url: str,
    api_key: str,
    model: str,
    segment_text: str,
    logger: logging.Logger,
) -> float | None:
    """Gemeinsamer HTTP-Aufruf fuer Groq und OpenRouter (beide
    OpenAI-kompatibel). Gibt None zurueck bei jedem Fehler (Netzwerk,
    Timeout, unerwartete Antwortstruktur) - niemals eine Exception."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(segment_text)},
        ],
        "temperature": 0.0,
        "max_tokens": 8,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
    except requests.RequestException as exc:
        logger.warning("LLM-Request fehlgeschlagen (%s) - Segment wird ohne LLM-Score bewertet.", exc)
        return None

    if response.status_code == 404 and "unavailable for free" in response.text.lower():
        logger.warning(
            "Konfiguriertes Modell '%s' ist nicht mehr kostenlos verfuegbar (HTTP 404). "
            "OpenRouters Free-Tier-Angebot wechselt haeufig - setze in config.yaml unter "
            "llm.openrouter_model entweder 'openrouter/free' (automatische Auswahl) oder "
            "ein aktuelles Modell von https://openrouter.ai/models?q=free. "
            "Segment wird ohne LLM-Score bewertet.",
            model,
        )
        return None

    if response.status_code != 200:
        logger.warning(
            "LLM-Request lieferte Status %d - Segment wird ohne LLM-Score bewertet: %s",
            response.status_code,
            response.text[:500],
        )
        return None

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        logger.warning("Unerwartete LLM-Antwortstruktur (%s) - Segment wird ohne LLM-Score bewertet.", exc)
        return None

    score = _parse_score_response(content)
    if score is None:
        logger.warning(
            "Konnte keine Zahl aus LLM-Antwort extrahieren (%r) - Segment wird ohne LLM-Score bewertet.",
            content,
        )
    return score


def score_segment_groq(text: str, api_key: str, model: str, logger: logging.Logger | None = None) -> float | None:
    """Bewertet ein einzelnes Segment ueber die Groq-API (Free-Tier)."""
    log = logger or logging.getLogger("autocut")
    return _call_chat_completions(_GROQ_URL, api_key, model, text, log)


def score_segment_openrouter(text: str, api_key: str, model: str, logger: logging.Logger | None = None) -> float | None:
    """Bewertet ein einzelnes Segment ueber die OpenRouter-API (Free-Modelle)."""
    log = logger or logging.getLogger("autocut")
    return _call_chat_completions(_OPENROUTER_URL, api_key, model, text, log)


def score_segments(
    transcript_segments: list[dict] | None,
    env: dict[str, str | None],
    config: LlmConfig,
    logger: logging.Logger | None = None,
) -> list[dict] | None:
    """Bewertet jedes Transkript-Segment per LLM und gibt eine Liste
    [{"start": float, "end": float, "score": float (0-10)}, ...] zurueck.

    Gibt None zurueck (mit Log-Hinweis, KEIN Fehler), wenn:
    - kein API_KEY in .env gesetzt ist
    - keine Transkript-Segmente vorhanden sind
    - der gewaehlte Provider unbekannt ist

    Segmente, bei denen der einzelne LLM-Request fehlschlaegt, werden
    aus dem Ergebnis ausgelassen (fuse_scores() faellt fuer diese
    Zeitfenster automatisch auf motion+audio zurueck).
    """
    log = logger or logging.getLogger("autocut")

    if not transcript_segments:
        log.debug("Keine Transkript-Segmente vorhanden - LLM-Scoring wird uebersprungen.")
        return None

    api_key = env.get("api_key")
    if not api_key:
        log.info(
            "Kein API_KEY in .env gefunden - LLM-Scoring wird automatisch uebersprungen "
            "(kein Fehler, Segmentauswahl basiert nur auf motion+audio)."
        )
        return None

    provider = (env.get("provider") or config.provider or "groq").lower()
    model = env.get("model") or (config.groq_model if provider == "groq" else config.openrouter_model)

    if provider == "groq":
        score_fn = score_segment_groq
    elif provider == "openrouter":
        score_fn = score_segment_openrouter
    else:
        log.warning(
            "Unbekannter LLM-Provider '%s' (erwartet 'groq' oder 'openrouter') - "
            "LLM-Scoring wird uebersprungen.",
            provider,
        )
        return None

    log.info(
        "Starte LLM-Segment-Scoring ueber '%s' (Modell: %s) fuer %d Segment(e) ...",
        provider,
        model,
        len(transcript_segments),
    )

    results: list[dict] = []
    for i, seg in enumerate(transcript_segments):
        score = score_fn(seg["text"], api_key, model, log)
        if score is not None:
            results.append({"start": seg["start"], "end": seg["end"], "score": score})
            log.info(
                "  %.0f/10 - \"%s\"",
                score,
                seg["text"][:80],
            )
        if i < len(transcript_segments) - 1 and config.request_delay_sec > 0:
            time.sleep(config.request_delay_sec)

    if not results:
        log.warning(
            "Kein einziges Segment konnte per LLM bewertet werden - "
            "Segmentauswahl basiert nur auf motion+audio."
        )
        return None

    log.info("LLM-Scoring abgeschlossen: %d von %d Segment(en) erfolgreich bewertet.", len(results), len(transcript_segments))
    return results
