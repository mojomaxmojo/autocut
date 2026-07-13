"""Video-Merge: fuegt mehrere Videos aus einem Ordner zu EINEM
durchgehenden Video zusammen, bevor die eigentliche Analyse-Pipeline
darauf laeuft (Erweiterung, auf Nutzerwunsch: "alle Videos in Videos/
sollen ein Stream werden und dann in der Verarbeitung landen").

Ablauf:
1. Alle Quell-Videos werden nacheinander (in Datei-Namen-Reihenfolge)
   zu einer einzigen Datei zusammengefuegt.
2. Zuerst wird versucht, das per ffmpeg concat-DEMUXER mit Stream-Copy
   zu tun (schnell, kein Qualitaetsverlust) - das funktioniert aber nur
   zuverlaessig, wenn alle Quell-Videos dieselbe Aufloesung/denselben
   Codec haben.
3. Schlaegt das fehl (z.B. weil die Videos von unterschiedlichen
   Kameras/Handys mit unterschiedlichen Aufloesungen stammen), wird
   automatisch auf den concat-FILTER zurueckgefallen, der jedes Video
   neu encodiert und auf eine einheitliche Aufloesung skaliert
   (langsamer, aber robust gegenueber gemischtem Quellmaterial).

Das zusammengefuegte Video landet im Cache-Verzeichnis
(.autocut_cache/_merged/<hash>/merged.mp4) und wird wie ein normales
einzelnes Eingabevideo an die bestehende Pipeline (run_analysis,
Score-Fusion, Export) weitergegeben - keine Aenderungen an den
bestehenden Analyse-/Export-Modulen noetig.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .checkpoint import merged_cache_dir
from .ffmpeg_utils import FfmpegError, run_ffmpeg


def _build_concat_list(video_paths: list[str], cache_dir: Path) -> str:
    """Erzeugt eine ffconcat-Datei mit absoluten Pfaden fuer den
    concat-Demuxer."""
    list_path = cache_dir / "merge_list.ffconcat"
    lines = ["ffconcat version 1.0"]
    for vp in video_paths:
        abs_path = Path(vp).resolve()
        lines.append(f"file '{abs_path}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(list_path)


def _try_stream_copy_merge(list_path: str, output_path: Path, logger: logging.Logger) -> bool:
    """Versucht das schnelle Stream-Copy-Merge (kein Re-Encode). Gibt
    True bei Erfolg zurueck, False wenn es fehlgeschlagen ist (dann
    faellt der Aufrufer auf den langsameren, robusteren Filter-Merge
    zurueck)."""
    try:
        run_ffmpeg(
            [
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                str(output_path),
            ],
            logger=logger,
        )
        return True
    except FfmpegError:
        logger.warning(
            "Schnelles Stream-Copy-Merge fehlgeschlagen (vermutlich unterschiedliche "
            "Aufloesungen/Codecs zwischen den Quell-Videos) - falle auf "
            "Re-Encode-Merge zurueck (langsamer, aber robuster)."
        )
        return False


def _filter_merge(video_paths: list[str], output_path: Path, logger: logging.Logger) -> None:
    """Robusteres Merge via concat-FILTER: skaliert alle Quell-Videos auf
    eine einheitliche Aufloesung (die des ERSTEN Videos) und encodiert
    neu. Notwendig, wenn die Quell-Videos unterschiedliche
    Aufloesungen/Codecs haben (z.B. verschiedene Kameras/Handys).
    """
    # Referenz-Aufloesung vom ersten Video ermitteln.
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_paths[0],
    ]
    result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    try:
        width_str, height_str = result.stdout.strip().split(",")
        width, height = int(width_str), int(height_str)
    except (ValueError, IndexError):
        width, height = 1920, 1080
        logger.warning("Konnte Referenz-Aufloesung nicht ermitteln, nutze Fallback 1920x1080.")

    inputs: list[str] = []
    filter_parts: list[str] = []
    for i, vp in enumerate(video_paths):
        inputs += ["-i", vp]
        filter_parts.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];"
            f"[{i}:a]aresample=async=1[a{i}]"
        )
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(video_paths)))
    filter_complex = ";".join(filter_parts) + f";{concat_inputs}concat=n={len(video_paths)}:v=1:a=1[outv][outa]"

    run_ffmpeg(
        [
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            str(output_path),
        ],
        logger=logger,
    )


def merge_videos(
    video_paths: list[str],
    cache_root: str,
    logger: logging.Logger | None = None,
) -> str:
    """Fuegt mehrere Videos zu EINEM durchgehenden Video zusammen und
    gibt den Pfad zur zusammengefuegten Datei zurueck.

    Checkpoint-geprueft: existiert die zusammengefuegte Datei bereits
    fuer exakt diese Menge an Quell-Dateien (Cache-Schluessel basiert
    auf Pfad+Groesse aller Quellen), wird sie wiederverwendet statt neu
    zu erzeugen.

    Reihenfolge: alphabetisch nach Dateiname (stabil, nachvollziehbar -
    z.B. "clip_01.mp4" vor "clip_02.mp4").
    """
    log = logger or logging.getLogger("autocut")

    if len(video_paths) == 1:
        log.info("Nur ein Video im Ordner gefunden - kein Merge notwendig.")
        return video_paths[0]

    sorted_paths = sorted(video_paths)
    cache_dir = merged_cache_dir(sorted_paths, cache_root)
    output_path = cache_dir / "merged.mp4"

    if output_path.exists() and output_path.stat().st_size > 0:
        log.info(
            "Zusammengefuegtes Video bereits vorhanden, ueberspringe Neuerstellung: %s",
            output_path,
        )
        return str(output_path)

    log.info(
        "Fuege %d Videos zu einem durchgehenden Stream zusammen (Reihenfolge: %s) ...",
        len(sorted_paths),
        ", ".join(Path(p).name for p in sorted_paths),
    )

    list_path = _build_concat_list(sorted_paths, cache_dir)
    if not _try_stream_copy_merge(list_path, output_path, log):
        _filter_merge(sorted_paths, output_path, log)

    log.info("Videos erfolgreich zusammengefuegt: %s", output_path)
    return str(output_path)
