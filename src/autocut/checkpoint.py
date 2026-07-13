"""Checkpointing-Helfer: jeder Pipeline-Schritt prueft, ob sein Ergebnis
bereits als JSON-Datei im Cache-Verzeichnis existiert, bevor er neu
rechnet. So kann die Pipeline nach einem Absturz fortgesetzt werden,
ohne alles neu zu berechnen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def video_cache_dir(video_path: str, cache_root: str) -> Path:
    """Gibt ein stabiles, pro-Video eindeutiges Cache-Verzeichnis zurueck,
    basierend auf dem absoluten Pfad + Dateigroesse (nicht dem Inhalt,
    da Hashing grosser Videodateien auf schwacher Hardware zu teuer
    waere).
    """
    abs_path = Path(video_path).resolve()
    try:
        size = abs_path.stat().st_size
    except FileNotFoundError:
        size = 0
    key = f"{abs_path}:{size}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    out_dir = Path(cache_root) / digest
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def merged_cache_dir(video_paths: list[str], cache_root: str) -> Path:
    """Wie video_cache_dir(), aber fuer eine Gruppe mehrerer Quelldateien
    (Video-Merge, siehe merge.py): der Cache-Schluessel basiert auf den
    sortierten (Pfad, Groesse)-Paaren aller Quell-Videos, damit derselbe
    Satz an Dateien immer denselben Cache-Ordner ergibt, unabhaengig von
    der Reihenfolge der Uebergabe.
    """
    entries = []
    for vp in video_paths:
        abs_path = Path(vp).resolve()
        try:
            size = abs_path.stat().st_size
        except FileNotFoundError:
            size = 0
        entries.append(f"{abs_path}:{size}")
    key = "|".join(sorted(entries))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    out_dir = Path(cache_root) / "_merged" / digest
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def checkpoint_exists(path: str | Path) -> bool:
    """Prueft, ob ein Checkpoint-JSON bereits existiert und lesbar ist."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        with p.open("r", encoding="utf-8") as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def write_checkpoint(path: str | Path, data: dict[str, Any]) -> None:
    """Schreibt einen Checkpoint als JSON-Datei (atomar via Temp-Datei +
    Rename, damit ein Absturz waehrend des Schreibens keine korrupte
    Checkpoint-Datei hinterlaesst)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(p.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(p)


def read_checkpoint(path: str | Path) -> dict[str, Any] | None:
    """Liest einen Checkpoint. Gibt None zurueck, wenn er nicht existiert
    oder beschaedigt ist (dann muss der aufrufende Code neu berechnen)."""
    p = Path(path)
    if not checkpoint_exists(p):
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
