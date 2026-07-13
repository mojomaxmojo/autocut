"""Konfiguration laden: config.yaml (Einstellungen) + .env (Secrets).

Keine hartcodierten Schwellenwerte im restlichen Code - alles kommt aus
diesen beiden Quellen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_ENV_PATH = ".env"


class ConfigError(Exception):
    """Wird geworfen, wenn config.yaml fehlt oder ungueltig ist."""


@dataclass
class Weights:
    motion: float = 0.2
    audio: float = 0.4
    llm: float = 0.4


@dataclass
class BeatsConfig:
    min_onsets_for_beat_mode: int = 4
    fallback_grid_interval_sec: float = 4.0
    # Maximaler Abstand (Sekunden), um den eine Schnittkante beim
    # Snapping verschoben werden darf. Liegt der naechste Snap-Punkt
    # weiter weg (z.B. weil Snap-Punkte in stillen Segmenten
    # herausgefiltert wurden und dadurch grosse Luecken entstehen),
    # bleibt die urspruengliche, ungesnappte Zeit erhalten - sonst
    # koennten Start und Ende eines Segments auf denselben Punkt
    # kollabieren.
    max_snap_distance_sec: float = 1.5


@dataclass
class MotionConfig:
    # Parameter fuer den ffmpeg mpdecimate-Filter (Frame-Verwurf bei
    # wenig Bewegung). Hoehere hi/lo-Werte = empfindlicher fuer Bewegung.
    hi: int = 768
    lo: int = 320
    frac: float = 0.33


@dataclass
class AudioConfig:
    # dB-Bereich zur Normalisierung des RMS-Pegels auf 0.0-1.0.
    floor_db: float = -60.0
    ceil_db: float = 0.0


@dataclass
class WhisperConfig:
    binary_path: str = "whisper.cpp/build/bin/whisper-cli"
    model_path: str = "whisper.cpp/models/ggml-small.bin"
    language: str = "de"
    # Gegenmassnahmen gegen bekannte whisper.cpp-Halluzinations-/
    # Wiederholungsschleifen (typisch bei Nicht-Sprache-Geraeuschen wie
    # Wind/Motor/Fahrradfahren): max_context=0 deaktiviert die
    # Kontextuebernahme zwischen Segmenten, ein hoeherer
    # entropy_threshold laesst das Modell bei unsicherer/sich
    # wiederholender Vorhersage eher neu ansetzen statt in der Schleife
    # zu bleiben. Siehe https://github.com/ggml-org/whisper.cpp/discussions/2286
    max_context: int = 0
    entropy_threshold: float = 2.6
    # Optionale VAD (Voice Activity Detection) - filtert Nicht-Sprache
    # VOR der Transkription heraus und reduziert Halluzinationen
    # zusaetzlich. Erfordert ein separates VAD-Modell (siehe README.md).
    # Leer lassen, um VAD zu deaktivieren.
    vad_model_path: str = ""


@dataclass
class LlmConfig:
    provider: str = "groq"
    groq_model: str = "llama-3.1-8b-instant"
    openrouter_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    request_delay_sec: float = 1.0


@dataclass
class OutputConfig:
    formats: list[str] = field(default_factory=lambda: ["16:9", "9:16", "1:1"])
    reel_lengths_sec: list[int] = field(default_factory=lambda: [60, 90, 120])
    clip_lengths_sec: list[int] = field(default_factory=lambda: [5, 10, 15])
    output_dir: str = "output"


@dataclass
class ResourcesConfig:
    max_parallel_jobs: int = 2
    ram_soft_limit_mb: int = 6000
    ram_warn_mb: int = 5000


@dataclass
class PathsConfig:
    cache_dir: str = ".autocut_cache"
    log_dir: str = "logs"


@dataclass
class Config:
    weights: Weights = field(default_factory=Weights)
    buckets_per_minute: float = 0.5
    proxy_resolution: int = 480
    silent_speed: int = 20
    bucket_window_sec: float = 5.0
    beats: BeatsConfig = field(default_factory=BeatsConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _dict_to_dataclass(data: dict[str, Any] | None, cls):
    if not data:
        return cls()
    valid_keys = {f for f in cls.__dataclass_fields__}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Config:
    """Laedt config.yaml und gibt ein validiertes Config-Objekt zurueck.

    Fehlt die Datei komplett, wird ein Fehler geworfen (Config ist
    Pflicht, damit keine versteckten Hardcoded-Werte einschleichen).
    Einzelne fehlende Felder in der YAML fallen auf sinnvolle Defaults
    zurueck.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(
            f"Konfigurationsdatei nicht gefunden: {path}. "
            "Lege eine config.yaml im Projektverzeichnis an "
            "(siehe mitgeliefertes Beispiel)."
        )

    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    weights = _dict_to_dataclass(raw.get("weights"), Weights)
    beats = _dict_to_dataclass(raw.get("beats"), BeatsConfig)
    motion = _dict_to_dataclass(raw.get("motion"), MotionConfig)
    audio = _dict_to_dataclass(raw.get("audio"), AudioConfig)
    whisper = _dict_to_dataclass(raw.get("whisper"), WhisperConfig)
    llm = _dict_to_dataclass(raw.get("llm"), LlmConfig)
    output = _dict_to_dataclass(raw.get("output"), OutputConfig)
    resources = _dict_to_dataclass(raw.get("resources"), ResourcesConfig)
    paths = _dict_to_dataclass(raw.get("paths"), PathsConfig)

    return Config(
        weights=weights,
        buckets_per_minute=raw.get("buckets_per_minute", 0.5),
        proxy_resolution=raw.get("proxy_resolution", 480),
        silent_speed=raw.get("silent_speed", 20),
        bucket_window_sec=raw.get("bucket_window_sec", 5.0),
        beats=beats,
        motion=motion,
        audio=audio,
        whisper=whisper,
        llm=llm,
        output=output,
        resources=resources,
        paths=paths,
    )


def load_env(path: str = DEFAULT_ENV_PATH) -> dict[str, str | None]:
    """Laedt .env (falls vorhanden) und gibt ein Dict mit den relevanten
    Secrets zurueck. Fehlt die .env komplett, wird NIEMALS ein Fehler
    geworfen - der KI-Scoring-Schritt faellt dann einfach automatisch
    weg (siehe llm_scoring.py in einem spaeteren Schritt).
    """
    values: dict[str, str | None] = {}
    if Path(path).exists():
        values = dict(dotenv_values(path))

    # Erlaubt auch echte Umgebungsvariablen als Override/Alternative zur .env
    api_key = os.environ.get("API_KEY") or values.get("API_KEY") or None
    provider = os.environ.get("API_PROVIDER") or values.get("API_PROVIDER") or "groq"
    model = os.environ.get("API_MODEL") or values.get("API_MODEL") or None

    return {
        "api_key": api_key,
        "provider": provider,
        "model": model,
    }
