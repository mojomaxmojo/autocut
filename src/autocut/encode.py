"""Encoding/Export (Schritt 5 aus FEATURE-PLAN.md): baut aus dem
Edit-Plan echte MP4-Highlight-Reels (mehrere Laengen x mehrere
Seitenverhaeltnisse) sowie automatisch abgeleitete Kurzclips.

Ablauf pro Reel-Laenge:
1. ffconcat-Datei mit ABSOLUTEN Pfaden erzeugen (robust aus jedem
   Arbeitsverzeichnis ausfuehrbar), die per ffmpeg concat-Demuxer die
   ausgewaehlten Segmente direkt aus der Originaldatei referenziert
   (inpoint/outpoint) - kein Vorab-Zuschneiden noetig.
2. Fuer jedes konfigurierte Seitenverhaeltnis (16:9/9:16/1:1) wird das
   Reel per ffmpeg neu encodiert (zentrierter Crop, keine Verzerrung).
   Neu-Encodieren ist hier notwendig, da die Segmente aus beliebigen
   Stellen der Originaldatei stammen (Stream-Copy waere nur an
   Keyframe-Grenzen moeglich, was die Schnittgenauigkeit ruinieren
   wuerde). Genutzt wird der erkannte Hardware-Encoder (VAAPI/QSV),
   mit automatischem Fallback auf libx264 bei einem Fehler.
3. Waehrend dieses Encodierens werden per -force_key_frames zusaetzliche
   Keyframes an den Vielfachen der kleinsten Kurzclip-Laenge erzwungen -
   dadurch kann die anschliessende Kurzclip-Aufteilung (Schritt "5.
   Kurzclips") per ffmpeg segment-Muxer mit STREAM-COPY (ohne
   Re-Encode) frame-genau an diesen Stellen schneiden.

Das Original-Rohmaterial bleibt in jedem Schritt unangetastet - alle
Ausgaben landen unter config.output.output_dir.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path

from .ffmpeg_utils import FfmpegError

_ASPECT_RATIOS: dict[str, float] = {
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "1:1": 1.0,
}


def _aspect_filename(aspect: str) -> str:
    return aspect.replace(":", "x")


def _gcd_of(values: list[int]) -> int:
    """Groesster gemeinsamer Teiler einer Liste (fuer den
    Keyframe-Intervall, damit alle konfigurierten Kurzclip-Laengen ohne
    Re-Encode exakt geschnitten werden koennen). Faellt auf 5 zurueck,
    wenn die Liste leer ist."""
    if not values:
        return 5
    result = values[0]
    for v in values[1:]:
        result = math.gcd(result, v)
    return max(1, result)


def _run(cmd: list[str], logger: logging.Logger) -> None:
    logger.debug("ffmpeg-Aufruf: %s", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        logger.error("ffmpeg-Fehler (Exit-Code %d): %s", result.returncode, result.stderr[-4000:])
        raise FfmpegError(f"ffmpeg-Aufruf fehlgeschlagen (Exit-Code {result.returncode})")


def build_ffconcat(
    edit_plan: list[tuple[float, float]],
    input_path: str,
    cache_dir: Path,
    name: str,
    logger: logging.Logger | None = None,
) -> str | None:
    """Erzeugt eine ffconcat-Datei mit absoluten Pfaden fuer den ffmpeg
    concat-Demuxer (`-f concat -safe 0`). Referenziert die Segmente per
    inpoint/outpoint direkt in der Originaldatei - kein Vorab-Zuschneiden.

    Gibt None zurueck, wenn der Edit-Plan leer ist (nichts zu exportieren).
    """
    log = logger or logging.getLogger("autocut")
    if not edit_plan:
        log.warning("Edit-Plan ist leer - kein ffconcat fuer '%s' erzeugt.", name)
        return None

    abs_input = str(Path(input_path).resolve())
    ffconcat_path = cache_dir / f"{name}.ffconcat"

    lines = ["ffconcat version 1.0"]
    for start, end in edit_plan:
        lines.append(f"file '{abs_input}'")
        lines.append(f"inpoint {start}")
        lines.append(f"outpoint {end}")

    ffconcat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.debug("ffconcat erzeugt: %s (%d Segment(e))", ffconcat_path, len(edit_plan))
    return str(ffconcat_path)


def _video_encoder_args(encoder: str) -> list[str]:
    if encoder == "qsv":
        return ["-c:v", "h264_qsv", "-preset", "medium"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]


def render_reel(
    ffconcat_path: str,
    output_path: str,
    hw_encoder: str,
    aspect_ratio: str,
    keyframe_interval: int,
    logger: logging.Logger | None = None,
) -> tuple[str, str]:
    """Rendert ein einzelnes Reel aus einer ffconcat-Datei, zugeschnitten
    auf das gewuenschte Seitenverhaeltnis (zentrierter Crop, keine
    Verzerrung/Letterboxing). Nutzt den erkannten Hardware-Encoder,
    faellt aber automatisch auf libx264 zurueck, falls VAAPI/QSV auf
    diesem System doch nicht funktioniert (z.B. Treiberproblem).

    Rueckgabe: (output_path, tatsaechlich_genutzter_encoder) - der
    Aufrufer kann den genutzten Encoder fuer nachfolgende Reels
    weiterverwenden, um wiederholte fehlschlagende Versuche zu vermeiden.
    """
    log = logger or logging.getLogger("autocut")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    ratio = _ASPECT_RATIOS.get(aspect_ratio, 16 / 9)
    # Zentrierter Crop auf das Ziel-Seitenverhaeltnis (kein Stretching).
    crop_filter = f"crop=w='min(iw,ih*{ratio})':h='min(ih,iw/{ratio})'"
    force_kf = f"expr:gte(t,n_forced*{keyframe_interval})"

    def _attempt(encoder: str) -> None:
        # WICHTIG: "-fflags +genpts" und "-avoid_negative_ts make_zero"
        # sind hier notwendig, weil die ffconcat-Datei mehrere
        # inpoint/outpoint-Segmente aus verschiedenen Stellen der
        # Originaldatei referenziert. Beim Zusammenfuegen entstehen
        # dadurch "rueckwaerts springende" Zeitstempel (Non-monotonic
        # DTS), die insbesondere den VAAPI-Treiber auf manchen Systemen
        # (z.B. Broadwell-Generation) zum Absturz bringen koennen.
        # "-fflags +genpts" laesst ffmpeg die Praesentationszeitstempel
        # neu und durchgehend berechnen statt die kaputten Original-DTS
        # zu uebernehmen.
        if encoder == "vaapi":
            vf = f"{crop_filter},format=nv12,hwupload"
            cmd = [
                "ffmpeg", "-hide_banner", "-y",
                "-fflags", "+genpts",
                "-vaapi_device", "/dev/dri/renderD128",
                "-f", "concat", "-safe", "0",
                "-i", ffconcat_path,
                "-vf", vf,
                "-c:v", "h264_vaapi",
                "-force_key_frames", force_kf,
                "-c:a", "aac", "-b:a", "128k",
                "-avoid_negative_ts", "make_zero",
                output_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-hide_banner", "-y",
                "-fflags", "+genpts",
                "-f", "concat", "-safe", "0",
                "-i", ffconcat_path,
                "-vf", crop_filter,
                *_video_encoder_args(encoder),
                "-force_key_frames", force_kf,
                "-c:a", "aac", "-b:a", "128k",
                "-avoid_negative_ts", "make_zero",
                output_path,
            ]
        _run(cmd, log)

    used_encoder = hw_encoder
    try:
        _attempt(hw_encoder)
    except FfmpegError:
        if hw_encoder == "libx264":
            raise
        log.warning(
            "Encoding mit '%s' fehlgeschlagen - falle auf libx264 (Software) zurueck.",
            hw_encoder,
        )
        used_encoder = "libx264"
        _attempt("libx264")

    log.info("Reel gerendert (%s): %s", used_encoder, output_path)
    return output_path, used_encoder


def split_into_clips(
    reel_path: str,
    clip_lengths: list[int],
    reel_output_dir: Path,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Erzeugt aus einem bereits gerenderten Reel automatisch Kurzclips
    fuer jede konfigurierte Laenge (z.B. 5/10/15s), via ffmpeg
    segment-Muxer mit STREAM-COPY (kein erneutes Encodieren noetig,
    da render_reel() bereits per -force_key_frames Keyframes an den
    passenden Vielfachen erzwungen hat).

    Ordnerstruktur: <reel_output_dir>/<length>s_000.mp4 usw.
    """
    log = logger or logging.getLogger("autocut")
    created: list[str] = []

    for length in clip_lengths:
        reel_output_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(reel_output_dir / f"{length}s_%03d.mp4")
        cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-i", reel_path,
            "-c", "copy",
            "-map", "0",
            "-f", "segment",
            "-segment_time", str(length),
            "-reset_timestamps", "1",
            pattern,
        ]
        try:
            _run(cmd, log)
        except FfmpegError:
            log.warning(
                "Kurzclip-Erstellung (%ds) via Stream-Copy fehlgeschlagen fuer %s - "
                "ueberspringe diese Laenge (kein Abbruch der Pipeline).",
                length,
                reel_path,
            )
            continue

        new_clips = sorted(reel_output_dir.glob(f"{length}s_*.mp4"))
        created.extend(str(p) for p in new_clips)
        log.info("Kurzclips (%ds) erzeugt: %d Datei(en) in %s", length, len(new_clips), reel_output_dir)

    return created


def export_all(
    edit_plans_by_length: dict[int, list[tuple[float, float]]],
    input_path: str,
    hw_encoder: str,
    cache_dir: Path,
    video_output_dir: Path,
    formats: list[str],
    clip_lengths: list[int],
    logger: logging.Logger | None = None,
) -> dict[str, list[str]]:
    """Orchestriert den kompletten Export fuer ein Video: fuer jede
    Reel-Laenge x jedes konfigurierte Seitenverhaeltnis wird ein Reel
    gerendert (reels/), danach werden aus dem jeweils ERSTEN
    konfigurierten Format (Standard: 16:9) automatisch die konfigurierten
    Kurzclip-Laengen abgeleitet (clips/), um die Anzahl der erzeugten
    Dateien auf schwacher Hardware nicht unnoetig zu vervielfachen.

    Rueckgabe: {"reels": [...Pfade...], "clips": [...Pfade...]}
    """
    log = logger or logging.getLogger("autocut")
    reels_dir = video_output_dir / "reels"
    clips_dir = video_output_dir / "clips"

    keyframe_interval = _gcd_of(clip_lengths)
    log.debug("Keyframe-Intervall fuer Kurzclip-Schnitte: %ds", keyframe_interval)

    active_encoder = hw_encoder
    all_reels: list[str] = []
    all_clips: list[str] = []
    primary_format = formats[0] if formats else "16:9"

    for reel_length, edit_plan in edit_plans_by_length.items():
        name = f"highlight_{reel_length}s"
        ffconcat_path = build_ffconcat(edit_plan, input_path, cache_dir, name, log)
        if ffconcat_path is None:
            log.warning("Ueberspringe Export fuer %ds-Reel (keine Segmente ausgewaehlt).", reel_length)
            continue

        primary_reel_path: str | None = None
        for aspect in formats:
            output_path = str(reels_dir / f"{name}_{_aspect_filename(aspect)}.mp4")
            try:
                rendered_path, active_encoder = render_reel(
                    ffconcat_path, output_path, active_encoder, aspect, keyframe_interval, log
                )
            except FfmpegError:
                log.error(
                    "Rendern des %ds-Reels (%s) ist endgueltig fehlgeschlagen - "
                    "ueberspringe dieses Format, Pipeline laeuft weiter.",
                    reel_length,
                    aspect,
                )
                continue
            all_reels.append(rendered_path)
            if aspect == primary_format:
                primary_reel_path = rendered_path

        if primary_reel_path and clip_lengths:
            clip_output_dir = clips_dir / name
            new_clips = split_into_clips(primary_reel_path, clip_lengths, clip_output_dir, log)
            all_clips.extend(new_clips)

    return {"reels": all_reels, "clips": all_clips}
