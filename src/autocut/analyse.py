"""Analyse-Grundlagen (Schritt 2 aus FEATURE-PLAN.md): Proxy-Encode,
Motion-Score (via ffmpeg mpdecimate) und Audio-Energie (via ffmpeg astats).

Alles laeuft komplett ohne KI und ist checkpoint-faehig: jeder Teilschritt
prueft zuerst, ob sein Ergebnis bereits im Cache-Verzeichnis liegt.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from .beats import get_snap_points
from .checkpoint import read_checkpoint, video_cache_dir, write_checkpoint
from .config import Config
from .ffmpeg_utils import get_avg_fps, probe_duration, run_ffmpeg
from .parallel import run_parallel
from .silence import merge_silence_with_snap, run_auto_editor

_SHOWINFO_PTS_RE = re.compile(r"pts_time:([0-9.]+)")
_AMETADATA_FRAME_RE = re.compile(r"pts_time:([0-9.]+)")
_AMETADATA_VALUE_RE = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(-?[0-9.]+|-inf|nan)")


@dataclass
class Bucket:
    """Ein Zeitfenster mit einem normalisierten Score (0.0-1.0)."""

    start: float
    end: float
    score: float


@dataclass
class AnalysisResult:
    input_path: str
    duration: float
    proxy_path: str
    cache_dir: str
    motion_buckets: list[Bucket] = field(default_factory=list)
    audio_buckets: list[Bucket] = field(default_factory=list)
    snap_points: list[float] = field(default_factory=list)
    silence_segments: list[dict] = field(default_factory=list)


def _buckets_to_json(buckets: list[Bucket]) -> list[dict]:
    return [{"start": b.start, "end": b.end, "score": b.score} for b in buckets]


def _buckets_from_json(data: list[dict]) -> list[Bucket]:
    return [Bucket(start=d["start"], end=d["end"], score=d["score"]) for d in data]


def make_proxy(
    input_path: str,
    cache_dir: Path,
    resolution: int,
    logger: logging.Logger | None = None,
) -> str:
    """Erzeugt ein kleines Proxy-Video (z.B. 480p) fuer die schnelle
    Analyse. Das Original bleibt dabei vollstaendig unangetastet.
    Checkpoint: existiert die Proxy-Datei bereits, wird nicht neu encodiert.
    """
    log = logger or logging.getLogger("autocut")
    proxy_path = cache_dir / "proxy.mp4"

    if proxy_path.exists() and proxy_path.stat().st_size > 0:
        log.info("Proxy bereits vorhanden, ueberspringe Neuerstellung: %s", proxy_path)
        return str(proxy_path)

    log.info("Erzeuge Proxy (%dp) fuer schnelle Analyse ...", resolution)
    run_ffmpeg(
        [
            "-i", input_path,
            "-vf", f"scale=-2:{resolution}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "30",
            "-c:a", "aac",
            "-b:a", "96k",
            str(proxy_path),
        ],
        logger=log,
    )
    log.info("Proxy erzeugt: %s", proxy_path)
    return str(proxy_path)


def _bucketize(
    timestamps: list[float],
    duration: float,
    window_sec: float,
    normalizer,
) -> list[Bucket]:
    """Teilt eine Liste von Zeitstempeln in gleich grosse Buckets und
    berechnet je Bucket einen normalisierten Score ueber `normalizer`
    (bekommt die Anzahl/Werte-Liste der Zeitstempel im Bucket)."""
    if duration <= 0:
        return []
    n_buckets = max(1, int(duration // window_sec) + 1)
    buckets: list[Bucket] = []
    for i in range(n_buckets):
        start = i * window_sec
        end = min(start + window_sec, duration)
        values_in_bucket = [t for t in timestamps if start <= t < end]
        score = normalizer(values_in_bucket, end - start)
        buckets.append(Bucket(start=round(start, 2), end=round(end, 2), score=round(score, 4)))
    return buckets


def motion_score(
    proxy_path: str,
    cache_dir: Path,
    duration: float,
    config: Config,
    logger: logging.Logger | None = None,
) -> list[Bucket]:
    """Berechnet einen Motion-Score pro Zeitfenster ueber den ffmpeg
    mpdecimate-Filter (NICHT scene-filter, wie gefordert).

    Prinzip: mpdecimate verwirft Frames, die sich kaum vom Vorgaenger
    unterscheiden. Kombiniert mit showinfo zaehlen wir, wie viele Frames
    pro Zeitfenster tatsaechlich "durchkommen" (=sich veraendert haben).
    Viele durchkommende Frames = viel Bewegung, wenige = wenig Bewegung.
    """
    log = logger or logging.getLogger("autocut")
    checkpoint_path = cache_dir / "motion.json"

    cached = read_checkpoint(checkpoint_path)
    if cached is not None:
        log.info("Motion-Score bereits vorhanden, ueberspringe Neuberechnung.")
        return _buckets_from_json(cached["buckets"])

    log.info("Berechne Motion-Score (mpdecimate) ...")
    mpdecimate_filter = (
        f"mpdecimate=hi={config.motion.hi}:lo={config.motion.lo}:frac={config.motion.frac},showinfo"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", proxy_path,
        "-vf", mpdecimate_filter,
        "-f", "null", "-",
    ]
    log.debug("ffmpeg-Aufruf (motion): %s", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        log.error("Motion-Analyse fehlgeschlagen (Exit-Code %d): %s", result.returncode, result.stderr[-2000:])
        log.warning("Fahre mit neutralem Motion-Score (0.5 ueberall) fort statt abzubrechen.")
        buckets = _bucketize([], duration, config.bucket_window_sec, lambda vals, w: 0.5)
    else:
        survived_timestamps = [float(m.group(1)) for m in _SHOWINFO_PTS_RE.finditer(result.stderr)]
        log.debug("Motion-Analyse: %d ueberlebende Frames gefunden", len(survived_timestamps))

        # Grobe Schaetzung der Proxy-Framerate fuer die Normalisierung.
        fps_estimate = get_avg_fps(proxy_path, log)

        def normalize(vals: list[float], window: float) -> float:
            if window <= 0 or fps_estimate <= 0:
                return 0.0
            expected_max = window * fps_estimate
            if expected_max <= 0:
                return 0.0
            return min(1.0, len(vals) / expected_max)

        buckets = _bucketize(survived_timestamps, duration, config.bucket_window_sec, normalize)

    write_checkpoint(checkpoint_path, {"buckets": _buckets_to_json(buckets)})
    log.info("Motion-Score berechnet: %d Zeitfenster", len(buckets))
    return buckets


def audio_energy(
    input_path: str,
    cache_dir: Path,
    duration: float,
    config: Config,
    logger: logging.Logger | None = None,
) -> list[Bucket]:
    """Berechnet einen Audio-Energie-Score pro Zeitfenster ueber den
    ffmpeg astats-Filter (RMS-Pegel). Erkennt laute/emotionale Momente
    auch ohne Musik (z.B. Wind, Wellen, Ausrufe), da rein auf Lautstaerke
    basierend, nicht auf Sprachinhalt.

    Laeuft direkt auf der Original-Audiospur (kein Proxy noetig, da
    Audio-Extraktion guenstig ist).
    """
    log = logger or logging.getLogger("autocut")
    checkpoint_path = cache_dir / "audio.json"

    cached = read_checkpoint(checkpoint_path)
    if cached is not None:
        log.info("Audio-Energie bereits vorhanden, ueberspringe Neuberechnung.")
        return _buckets_from_json(cached["buckets"])

    log.info("Berechne Audio-Energie (astats) ...")
    af_filter = (
        "astats=metadata=1:reset=1,"
        "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", input_path,
        "-af", af_filter,
        "-f", "null", "-",
    ]
    log.debug("ffmpeg-Aufruf (audio): %s", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        log.error("Audio-Analyse fehlgeschlagen (Exit-Code %d): %s", result.returncode, result.stderr[-2000:])
        log.warning("Fahre mit neutralem Audio-Score (0.5 ueberall) fort statt abzubrechen.")
        buckets = _bucketize([], duration, config.bucket_window_sec, lambda vals, w: 0.5)
    else:
        samples = _parse_astats_output(result.stderr)
        log.debug("Audio-Analyse: %d RMS-Messpunkte gefunden", len(samples))
        buckets = _bucketize_audio_samples(samples, duration, config)

    write_checkpoint(checkpoint_path, {"buckets": _buckets_to_json(buckets)})
    log.info("Audio-Energie berechnet: %d Zeitfenster", len(buckets))
    return buckets


def _parse_astats_output(stderr_output: str) -> list[tuple[float, float]]:
    """Parst die ametadata=print-Ausgabe von astats zu einer Liste von
    (pts_time, rms_db) Paaren. Frames ohne gueltigen RMS-Wert (z.B. -inf
    bei absoluter Stille) werden als sehr leise (-90 dB) gewertet."""
    samples: list[tuple[float, float]] = []
    current_pts: float | None = None
    for line in stderr_output.splitlines():
        pts_match = _AMETADATA_FRAME_RE.search(line)
        if pts_match and "frame:" in line:
            current_pts = float(pts_match.group(1))
            continue
        value_match = _AMETADATA_VALUE_RE.search(line)
        if value_match and current_pts is not None:
            raw = value_match.group(1)
            if raw in ("-inf", "nan"):
                rms_db = -90.0
            else:
                rms_db = float(raw)
            samples.append((current_pts, rms_db))
    return samples


def _bucketize_audio_samples(
    samples: list[tuple[float, float]],
    duration: float,
    config: Config,
) -> list[Bucket]:
    if duration <= 0:
        return []
    window_sec = config.bucket_window_sec
    n_buckets = max(1, int(duration // window_sec) + 1)
    floor_db = config.audio.floor_db
    ceil_db = config.audio.ceil_db
    span = max(1e-6, ceil_db - floor_db)

    buckets: list[Bucket] = []
    for i in range(n_buckets):
        start = i * window_sec
        end = min(start + window_sec, duration)
        values = [db for (t, db) in samples if start <= t < end]
        if values:
            avg_db = sum(values) / len(values)
        else:
            avg_db = floor_db
        normalized = (avg_db - floor_db) / span
        score = max(0.0, min(1.0, normalized))
        buckets.append(Bucket(start=round(start, 2), end=round(end, 2), score=round(score, 4)))
    return buckets


def run_analysis(
    input_path: str,
    config: Config,
    logger: logging.Logger | None = None,
) -> AnalysisResult:
    """Orchestriert Proxy-Erstellung + Motion-Score + Audio-Energie fuer
    ein einzelnes Video, mit Checkpointing pro Teilschritt.
    """
    log = logger or logging.getLogger("autocut")

    cache_dir = video_cache_dir(input_path, config.paths.cache_dir)
    log.info("Cache-Verzeichnis fuer dieses Video: %s", cache_dir)

    duration = probe_duration(input_path, log)
    log.info("Videolaenge: %.1f Sekunden", duration)

    proxy_path = make_proxy(input_path, cache_dir, config.proxy_resolution, log)

    # Motion-Score, Audio-Energie, Beat-Erkennung (aubio) und
    # Stille-Erkennung (auto-editor) sind alle unabhaengig voneinander
    # und laufen daher parallel (begrenzt durch resources.max_parallel_jobs,
    # damit der Dual-Core-CPU nicht ueberlastet wird).
    tasks = [
        partial(motion_score, proxy_path, cache_dir, duration, config, log),
        partial(audio_energy, input_path, cache_dir, duration, config, log),
        partial(get_snap_points, input_path, duration, config, cache_dir, log),
        partial(run_auto_editor, input_path, cache_dir, config.silent_speed, log),
    ]
    motion_buckets, audio_buckets, snap_points, silence_segments = run_parallel(
        tasks, max_workers=config.resources.max_parallel_jobs, logger=log
    )

    snap_points = merge_silence_with_snap(snap_points, silence_segments)

    return AnalysisResult(
        input_path=input_path,
        duration=duration,
        proxy_path=proxy_path,
        cache_dir=str(cache_dir),
        motion_buckets=motion_buckets,
        audio_buckets=audio_buckets,
        snap_points=snap_points,
        silence_segments=silence_segments,
    )
