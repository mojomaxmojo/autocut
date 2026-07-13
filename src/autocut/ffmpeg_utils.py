"""Zentrale ffmpeg/ffprobe-Helfer: Hardware-Encoder-Erkennung, Dauer-Abfrage
und ein einheitlicher subprocess-Wrapper mit Logging.

Es wird bewusst kein ffmpeg-python o.ae. genutzt - direkte subprocess-Aufrufe
sind auf schwacher Hardware transparenter und leichter zu debuggen.
"""

from __future__ import annotations

import logging
import shutil
import subprocess


class FfmpegError(RuntimeError):
    """Wird geworfen, wenn ein ffmpeg/ffprobe-Aufruf fehlschlaegt."""


def run_ffmpeg(
    args: list[str],
    logger: logging.Logger | None = None,
    capture_stdout: bool = False,
) -> subprocess.CompletedProcess:
    """Fuehrt einen ffmpeg-Befehl aus (args OHNE das fuehrende 'ffmpeg').

    Bei einem Fehler (return code != 0) wird stderr geloggt und ein
    FfmpegError geworfen - der aufrufende Code entscheidet dann ueber
    einen Fallback (z.B. neutraler Score statt Absturz).
    """
    log = logger or logging.getLogger("autocut")
    cmd = ["ffmpeg", "-hide_banner", "-y", *args]
    log.debug("ffmpeg-Aufruf: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        log.error("ffmpeg-Fehler (Exit-Code %d): %s", result.returncode, result.stderr[-4000:])
        raise FfmpegError(f"ffmpeg-Aufruf fehlgeschlagen (Exit-Code {result.returncode})")
    return result


def probe_duration(path: str, logger: logging.Logger | None = None) -> float:
    """Gibt die Videolaenge in Sekunden zurueck (via ffprobe)."""
    log = logger or logging.getLogger("autocut")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        log.error("ffprobe-Fehler fuer %s: %s", path, result.stderr)
        raise FfmpegError(f"Konnte Videolaenge nicht ermitteln: {path}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise FfmpegError(f"Unerwartete ffprobe-Ausgabe fuer {path}: {result.stdout!r}") from exc


def detect_hw_encoder(logger: logging.Logger | None = None) -> str:
    """Erkennt, welcher Hardware-Encoder auf diesem System verfuegbar ist.

    Rueckgabe: "vaapi", "qsv" oder "libx264" (Fallback). Wird beim ersten
    Aufruf einmal geloggt, damit klar ist, welcher Pfad aktiv ist.
    Diese Funktion entscheidet nur, WELCHER Encoder genutzt werden
    koennte - der tatsaechliche Encode-Aufruf folgt erst in Schritt 5.
    """
    log = logger or logging.getLogger("autocut")

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        encoders_output = result.stdout
    except FileNotFoundError:
        log.error("ffmpeg wurde nicht gefunden - bitte installieren (siehe README.md).")
        raise FfmpegError("ffmpeg nicht installiert")

    has_vaapi = "h264_vaapi" in encoders_output
    has_qsv = "h264_qsv" in encoders_output

    vaapi_usable = False
    if has_vaapi and shutil.which("vainfo"):
        vainfo_result = subprocess.run(
            ["vainfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        vaapi_usable = "VAEntrypoint" in vainfo_result.stdout

    if vaapi_usable:
        log.info("Hardware-Encoder erkannt: VAAPI (Intel iGPU)")
        return "vaapi"
    if has_qsv:
        log.info("Hardware-Encoder erkannt: Intel QSV")
        return "qsv"

    log.info(
        "Kein nutzbarer Hardware-Encoder gefunden (VAAPI/QSV) - "
        "Fallback auf Software-Encoder libx264."
    )
    return "libx264"
