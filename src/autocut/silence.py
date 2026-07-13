"""Stille-Erkennung/Grobschnitt via auto-editor (Schritt 3 aus
FEATURE-PLAN.md).

auto-editor markiert/beschleunigt stille Passagen, statt sie hart
herauszuschneiden (silent_speed, Standard 16-30x statt hartem Cut). Wir
nutzen den JSON-Timeline-Export von auto-editor, um die Zeitfenster zu
bekommen, die als "still" (= beschleunigt) markiert wurden - diese
Information dient in Schritt 4 dazu, Highlight-Segmente NICHT in
stille Passagen zu legen bzw. Schnittkanten mit den Beat-Snap-Punkten
zu verschmelzen.

WICHTIG: Die auto-editor CLI-Flags haben sich zwischen Versionen leicht
veraendert. Diese Funktion ist deshalb bewusst defensiv geschrieben:
schlaegt der Aufruf fehl oder laesst sich die Ausgabe nicht parsen, wird
NUR eine Warnung geloggt und eine leere Liste zurueckgegeben - die
Pipeline laeuft dann einfach ohne Stille-Information weiter (identisches
Fallback-Prinzip wie bei den ffmpeg-Analysen in analyse.py).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from .checkpoint import read_checkpoint, write_checkpoint


def _auto_editor_available(logger: logging.Logger) -> bool:
    if shutil.which("auto-editor") is None:
        logger.warning(
            "auto-editor wurde nicht gefunden - Stille-Erkennung wird "
            "uebersprungen (kein Fehler, Pipeline laeuft normal weiter). "
            "Installation z.B. via 'pipx install auto-editor', siehe README.md."
        )
        return False
    return True


def run_auto_editor(
    input_path: str,
    cache_dir: Path,
    silent_speed: float,
    logger: logging.Logger | None = None,
) -> list[dict]:
    """Ruft auto-editor auf, um stille Passagen zu erkennen, und gibt eine
    Liste von Segmenten zurueck: [{"start": float, "end": float, "silent": bool}, ...]

    Checkpoint-geprueft (silence.json im Cache-Verzeichnis). Bei jedem
    Fehler (Binary fehlt, Aufruf schlaegt fehl, Ausgabe nicht parsbar)
    wird eine leere Liste zurueckgegeben statt einer Exception.
    """
    log = logger or logging.getLogger("autocut")
    checkpoint_path = cache_dir / "silence.json"

    cached = read_checkpoint(checkpoint_path)
    if cached is not None:
        log.info("Stille-Analyse bereits vorhanden, ueberspringe Neuberechnung.")
        return cached["segments"]

    if not _auto_editor_available(log):
        write_checkpoint(checkpoint_path, {"segments": []})
        return []

    timeline_path = cache_dir / "auto_editor_timeline.json"
    cmd = [
        "auto-editor",
        input_path,
        "--edit", "audio",
        "--silent-speed", str(silent_speed),
        "--video-speed", "1",
        "--export", "json",
        "--output", str(timeline_path),
        "--no-open",
    ]
    log.info("Starte auto-editor Stille-Analyse (silent_speed=%s) ...", silent_speed)
    log.debug("auto-editor-Aufruf: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
    except OSError as exc:
        log.warning("auto-editor konnte nicht ausgefuehrt werden (%s) - ueberspringe.", exc)
        write_checkpoint(checkpoint_path, {"segments": []})
        return []

    if result.returncode != 0:
        log.warning(
            "auto-editor ist fehlgeschlagen (Exit-Code %d): %s - "
            "ueberspringe Stille-Analyse, Pipeline laeuft ohne diese Information weiter.",
            result.returncode,
            (result.stderr or result.stdout)[-1500:],
        )
        write_checkpoint(checkpoint_path, {"segments": []})
        return []

    segments = _parse_timeline(timeline_path, silent_speed, log)
    write_checkpoint(checkpoint_path, {"segments": segments})
    log.info("Stille-Analyse abgeschlossen: %d Segment(e) erkannt.", len(segments))
    return segments


def _parse_timeline(timeline_path: Path, silent_speed: float, logger: logging.Logger) -> list[dict]:
    """Parst den JSON-Timeline-Export von auto-editor. Robust gegenueber
    leicht unterschiedlichen Strukturen zwischen auto-editor-Versionen -
    fehlt eine erwartete Struktur, wird einfach eine leere Liste
    zurueckgegeben statt abzustuerzen.
    """
    if not timeline_path.exists():
        logger.warning(
            "auto-editor hat keine Timeline-Datei erzeugt (%s) - "
            "ueberspringe Stille-Analyse.",
            timeline_path,
        )
        return []

    try:
        with timeline_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Konnte auto-editor Timeline nicht lesen (%s) - ueberspringe.", exc)
        return []

    try:
        timebase = data.get("timebase", "30/1")
        num, _, denom = timebase.partition("/")
        fps = float(num) / float(denom) if denom else float(num)

        chunks = None
        if "v" in data and isinstance(data["v"], list) and data["v"]:
            chunks = data["v"][0].get("chunks")
        if chunks is None:
            chunks = data.get("chunks")
        if not chunks:
            logger.warning(
                "Unerwartetes auto-editor Timeline-Format (keine 'chunks' gefunden) - "
                "ueberspringe Stille-Analyse."
            )
            return []

        segments = []
        for chunk in chunks:
            start_frame, end_frame, speed = chunk[0], chunk[1], chunk[2]
            is_silent = float(speed) > 1.0
            segments.append(
                {
                    "start": round(start_frame / fps, 2),
                    "end": round(end_frame / fps, 2),
                    "silent": is_silent,
                }
            )
        return segments
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        logger.warning(
            "Konnte auto-editor Timeline nicht interpretieren (%s) - "
            "ueberspringe Stille-Analyse, Pipeline laeuft normal weiter.",
            exc,
        )
        return []


def merge_silence_with_snap(snap_points: list[float], silence_segments: list[dict]) -> list[float]:
    """Entfernt Snap-Punkte, die mitten in einem als 'silent' markierten
    Segment liegen (reine Datenverarbeitung, kein I/O). So werden
    Schnittkanten in Schritt 4 nicht in stille Passagen gelegt, wenn
    Alternativen ausserhalb der Stille verfuegbar sind.

    Bleiben nach dem Filtern keine Snap-Punkte mehr uebrig (z.B. wenn
    das gesamte Video als "still" markiert wurde), werden die
    urspruenglichen Snap-Punkte unveraendert zurueckgegeben - lieber
    etwas ungenauer snappen als gar keine Snap-Punkte mehr zu haben.
    """
    if not silence_segments:
        return snap_points

    def in_silent_segment(t: float) -> bool:
        return any(seg["silent"] and seg["start"] <= t < seg["end"] for seg in silence_segments)

    filtered = [p for p in snap_points if not in_silent_segment(p)]
    return filtered if filtered else snap_points
