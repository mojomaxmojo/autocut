"""Beat/Onset-Erkennung (Schritt 3 aus FEATURE-PLAN.md).

Nutzt aubio (aubioonset) um Zeitstempel zu finden, an denen sich das
Audiosignal deutlich veraendert (Onsets). Bei Reise-/Landschaftsvideos
gibt es meist keine klare Musik mit Takt - deshalb gibt es einen
verlaesslichen Fallback auf ein gleichmaessiges Zeitraster, wenn zu
wenige Onsets gefunden werden.

Diese Zeitpunkte dienen spaeter (Schritt 4) als "Snap-Punkte", auf die
Schnittkanten gezogen werden, damit Schnitte nicht mitten in eine
Bewegung/einen Ton fallen.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from .checkpoint import read_checkpoint, write_checkpoint
from .config import Config


def _aubio_available(logger: logging.Logger) -> bool:
    if shutil.which("aubioonset") is None:
        logger.warning(
            "aubioonset (aus dem Paket 'aubio') wurde nicht gefunden - "
            "nutze direkt den Zeitraster-Fallback fuer Beat/Pause-Snapping."
        )
        return False
    return True


def detect_beats(
    input_path: str,
    logger: logging.Logger | None = None,
) -> list[float]:
    """Ruft aubioonset per subprocess auf und gibt die erkannten
    Onset-Zeitstempel (in Sekunden) zurueck. Gibt eine leere Liste
    zurueck, wenn aubioonset fehlt oder fehlschlaegt - niemals eine
    Exception, damit der Fallback in get_snap_points() sauber greift.
    """
    log = logger or logging.getLogger("autocut")

    if not _aubio_available(log):
        return []

    cmd = ["aubioonset", "-i", input_path]
    log.debug("aubio-Aufruf: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
    except OSError as exc:
        log.warning("aubioonset konnte nicht ausgefuehrt werden (%s) - nutze Fallback.", exc)
        return []

    if result.returncode != 0:
        log.warning(
            "aubioonset ist fehlgeschlagen (Exit-Code %d): %s - nutze Fallback.",
            result.returncode,
            result.stderr[-1000:],
        )
        return []

    onsets: list[float] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            onsets.append(float(line))
        except ValueError:
            continue

    log.debug("aubio hat %d Onset(s) gefunden", len(onsets))
    return onsets


def fallback_grid(duration: float, interval_sec: float) -> list[float]:
    """Erzeugt ein gleichmaessiges Zeitraster als Snap-Punkte, wenn keine
    verlaesslichen Onsets erkannt wurden (z.B. Video ohne Musik)."""
    if duration <= 0 or interval_sec <= 0:
        return [0.0]
    points = []
    t = 0.0
    while t < duration:
        points.append(round(t, 2))
        t += interval_sec
    return points


def get_snap_points(
    input_path: str,
    duration: float,
    config: Config,
    cache_dir: Path,
    logger: logging.Logger | None = None,
) -> list[float]:
    """Kombiniert aubio-Onset-Erkennung mit Zeitraster-Fallback und liefert
    eine sortierte Liste von Zeitpunkten (Sekunden), auf die Schnittkanten
    spaeter gesnapped werden koennen. Checkpoint-geprueft.
    """
    log = logger or logging.getLogger("autocut")
    checkpoint_path = cache_dir / "beats.json"

    cached = read_checkpoint(checkpoint_path)
    if cached is not None:
        log.info("Beat/Snap-Punkte bereits vorhanden, ueberspringe Neuberechnung.")
        return cached["snap_points"]

    onsets = detect_beats(input_path, log)

    if len(onsets) >= config.beats.min_onsets_for_beat_mode:
        log.info("aubio: %d Onsets erkannt - nutze Beat-Modus fuer Snap-Punkte.", len(onsets))
        snap_points = sorted(onsets)
        mode = "beats"
    else:
        log.info(
            "aubio: nur %d Onset(s) erkannt (Schwellenwert: %d) - "
            "zu wenige fuer klaren Takt, nutze Zeitraster-Fallback (alle %.1fs).",
            len(onsets),
            config.beats.min_onsets_for_beat_mode,
            config.beats.fallback_grid_interval_sec,
        )
        snap_points = fallback_grid(duration, config.beats.fallback_grid_interval_sec)
        mode = "fallback_grid"

    write_checkpoint(checkpoint_path, {"snap_points": snap_points, "mode": mode})
    log.info("Snap-Punkte bereit: %d (Modus: %s)", len(snap_points), mode)
    return snap_points
