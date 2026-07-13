"""Lokale Transkription via whisper.cpp (Schritt 6 aus FEATURE-PLAN.md,
optionaler KI-Layer - nur aktiv wenn --no-ai NICHT gesetzt ist).

Ablauf:
1. Audio aus dem Original-Video als 16kHz-Mono-WAV extrahieren (das ist
   das von whisper.cpp erwartete Eingabeformat).
2. whisper.cpp (whisper-cli) mit dem konfigurierten (kleinen) Modell und
   Sprache "de" aufrufen, Ausgabe als JSON (--output-json) fuer robustes
   Parsing der Zeitstempel-Segmente.
3. Segmente mit Start/Ende (Sekunden) und Text zurueckgeben.

Fehlt die whisper.cpp-Binary oder das Modell, oder schlaegt der Aufruf
fehl, wird IMMER None zurueckgegeben (nie eine Exception) - die
Pipeline laeuft dann exakt wie im --no-ai Modus weiter, nur mit einer
Log-Warnung. Der KI-Layer ist ein optionaler Zusatz, kein
Kernbestandteil.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from .checkpoint import read_checkpoint, write_checkpoint
from .config import WhisperConfig
from .ffmpeg_utils import FfmpegError, run_ffmpeg


def whisper_available(config: WhisperConfig, logger: logging.Logger | None = None) -> bool:
    """Prueft, ob die konfigurierte whisper.cpp-Binary und das Modell
    tatsaechlich existieren und die Binary ausfuehrbar ist."""
    log = logger or logging.getLogger("autocut")

    binary_path = Path(config.binary_path)
    model_path = Path(config.model_path)

    if not binary_path.exists():
        log.warning(
            "whisper.cpp-Binary nicht gefunden unter '%s' - Transkription wird "
            "uebersprungen (kein Fehler). Siehe README.md fuer die Installation.",
            binary_path,
        )
        return False
    if not model_path.exists():
        log.warning(
            "whisper.cpp-Modell nicht gefunden unter '%s' - Transkription wird "
            "uebersprungen (kein Fehler). Modell z.B. via "
            "'bash whisper.cpp/models/download-ggml-model.sh small' laden.",
            model_path,
        )
        return False
    return True


def _extract_audio_for_whisper(input_path: str, cache_dir: Path, logger: logging.Logger) -> str | None:
    """Extrahiert die Audiospur als 16kHz-Mono-WAV (von whisper.cpp
    erwartetes Format). Gibt None zurueck, wenn die Extraktion
    fehlschlaegt (z.B. Video ohne Audiospur)."""
    wav_path = cache_dir / "audio_for_whisper.wav"
    if wav_path.exists() and wav_path.stat().st_size > 0:
        logger.info("Audio fuer Transkription bereits extrahiert, ueberspringe: %s", wav_path)
        return str(wav_path)

    try:
        run_ffmpeg(
            [
                "-i", input_path,
                "-vn",
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(wav_path),
            ],
            logger=logger,
        )
    except FfmpegError:
        logger.warning(
            "Audio-Extraktion fuer Transkription fehlgeschlagen (z.B. keine Audiospur) - "
            "ueberspringe Transkription."
        )
        return None
    return str(wav_path)


def _parse_whisper_json(json_path: Path, logger: logging.Logger) -> list[dict]:
    """Parst die --output-json Ausgabe von whisper.cpp
    ({"transcription": [{"offsets": {"from": ms, "to": ms}, "text": "..."}]})
    zu einer Liste von {"start": float, "end": float, "text": str}
    (start/end in Sekunden). Robust gegenueber fehlenden/leeren Dateien."""
    if not json_path.exists():
        logger.warning("whisper.cpp hat keine JSON-Ausgabe erzeugt (%s) - ueberspringe.", json_path)
        return []

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Konnte whisper.cpp JSON-Ausgabe nicht lesen (%s) - ueberspringe.", exc)
        return []

    entries = data.get("transcription", [])
    segments = []
    for entry in entries:
        try:
            offsets = entry["offsets"]
            text = entry.get("text", "").strip()
            if not text:
                continue
            segments.append(
                {
                    "start": round(offsets["from"] / 1000.0, 2),
                    "end": round(offsets["to"] / 1000.0, 2),
                    "text": text,
                }
            )
        except (KeyError, TypeError):
            continue
    return segments


def transcribe(
    input_path: str,
    config: WhisperConfig,
    cache_dir: Path,
    logger: logging.Logger | None = None,
) -> list[dict] | None:
    """Transkribiert das Audio eines Videos lokal via whisper.cpp.

    Gibt eine Liste von Segmenten [{"start": float, "end": float, "text": str}, ...]
    zurueck, oder None, wenn whisper.cpp nicht verfuegbar ist oder der
    Aufruf fehlschlaegt (niemals eine Exception - Kernprinzip: die
    Pipeline muss auch ohne diesen KI-Schritt vollstaendig funktionieren).

    Checkpoint-geprueft (transcript.json im Cache-Verzeichnis).
    """
    log = logger or logging.getLogger("autocut")
    checkpoint_path = cache_dir / "transcript.json"

    cached = read_checkpoint(checkpoint_path)
    if cached is not None:
        log.info("Transkript bereits vorhanden, ueberspringe Neuberechnung.")
        return cached["segments"] if cached["segments"] else None

    # WICHTIG: "nicht verfuegbar" (Binary/Modell fehlt, keine Audiospur)
    # wird bewusst NICHT gecacht. Sonst wuerde ein spaeterer Lauf, bei
    # dem whisper.cpp inzwischen korrekt installiert wurde, faelschlich
    # den alten "nicht verfuegbar"-Checkpoint wiederverwenden und nie
    # wieder eine echte Transkription versuchen. Nur ein tatsaechlich
    # DURCHGEFUEHRTER Versuch (erfolgreich oder mit ffmpeg/whisper.cpp-
    # Laufzeitfehler) wird als Checkpoint persistiert.
    if not whisper_available(config, log):
        return None

    wav_path = _extract_audio_for_whisper(input_path, cache_dir, log)
    if wav_path is None:
        return None

    output_prefix = cache_dir / "transcript"
    cmd = [
        config.binary_path,
        "-m", config.model_path,
        "-f", wav_path,
        "-l", config.language,
        "--output-json",
        "--output-file", str(output_prefix),
        "--no-prints",
    ]
    log.info("Starte whisper.cpp-Transkription (Sprache: %s, Modell: %s) ...", config.language, config.model_path)
    log.debug("whisper.cpp-Aufruf: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
    except OSError as exc:
        log.warning("whisper.cpp konnte nicht ausgefuehrt werden (%s) - ueberspringe Transkription.", exc)
        # Kein Checkpoint bei Laufzeitfehlern - ein erneuter Versuch
        # koennte spaeter erfolgreich sein (z.B. nach Beheben eines
        # RAM-Engpasses auf schwacher Hardware).
        return None

    if result.returncode != 0:
        log.warning(
            "whisper.cpp ist fehlgeschlagen (Exit-Code %d): %s - "
            "ueberspringe Transkription, Pipeline laeuft ohne KI-Layer weiter.",
            result.returncode,
            (result.stderr or result.stdout)[-1500:],
        )
        return None

    json_path = Path(f"{output_prefix}.json")
    segments = _parse_whisper_json(json_path, log)

    write_checkpoint(checkpoint_path, {"segments": segments})
    log.info("Transkription abgeschlossen: %d Segment(e) erkannt.", len(segments))
    return segments if segments else None
