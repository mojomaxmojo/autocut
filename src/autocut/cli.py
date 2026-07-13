"""CLI-Einstiegspunkt fuer das Autocut Highlight Tool.

Laedt Konfiguration + .env, richtet Logging und RAM-Schutz ein und
fuehrt die komplette Pipeline aus: Proxy-Encode, Motion-Score,
Audio-Energie, Beat/Stille-Erkennung (Schritt 2/3), Score-Fusion und
Segmentauswahl mit Snap-to-Beat (Schritt 4), sowie den Video-Export als
Highlight-Reels + Kurzclips in mehreren Formaten (Schritt 5). Optionale
KI-Schritte (Transkription, LLM-Scoring) werden gemaess
FEATURE-PLAN.md ergaenzt.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from .analyse import run_analysis
from .checkpoint import video_cache_dir
from .config import Config, ConfigError, load_config, load_env
from .encode import export_all
from .ffmpeg_utils import detect_hw_encoder
from .logging_setup import setup_logging
from .resources import set_soft_ram_limit, warn_if_high_memory
from .scoring import build_edit_plan, fuse_scores, select_buckets


def _parse_int_list(value: str) -> list[int]:
    """Parst ein CLI-Argument wie '60,90,120' zu [60, 90, 120]."""
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _print_summary(
    logger: logging.Logger,
    input_path: str,
    config: Config,
    no_ai: bool,
    reel_lengths: list[int],
    clip_lengths: list[int],
    env: dict[str, str | None],
) -> None:
    logger.info("=" * 60)
    logger.info("Autocut Highlight CLI - Konfigurationsuebersicht")
    logger.info("=" * 60)
    logger.info("Input:                %s", input_path)
    logger.info("Modus:                %s", "OHNE KI (--no-ai)" if no_ai else "MIT KI (falls verfuegbar)")
    logger.info(
        "Gewichte (motion/audio/llm): %.2f / %.2f / %.2f",
        config.weights.motion,
        config.weights.audio,
        config.weights.llm,
    )
    logger.info("Buckets pro Minute:   %.2f", config.buckets_per_minute)
    logger.info("Proxy-Aufloesung:     %dp", config.proxy_resolution)
    logger.info("Silent-Speed:         %dx", config.silent_speed)
    logger.info("Reel-Laengen (s):     %s", reel_lengths or config.output.reel_lengths_sec)
    logger.info("Clip-Laengen (s):     %s", clip_lengths or config.output.clip_lengths_sec)
    logger.info("Formate:              %s", ", ".join(config.output.formats))
    logger.info("Max. parallele Jobs:  %d", config.resources.max_parallel_jobs)

    if not no_ai:
        if env.get("api_key"):
            logger.info("LLM-Provider:         %s (API_KEY gefunden)", env.get("provider"))
        else:
            logger.info(
                "LLM-Provider:         kein API_KEY in .env gefunden -> "
                "LLM-Scoring wird automatisch uebersprungen (kein Fehler)."
            )
    logger.info("=" * 60)


@click.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Pfad zu einer Video-Datei oder einem Ordner mit mehreren Videos.",
)
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    show_default=True,
    help="Pfad zur config.yaml.",
)
@click.option(
    "--no-ai",
    is_flag=True,
    default=False,
    help="Deaktiviert alle KI-Schritte (Transkription + LLM-Scoring) komplett.",
)
@click.option(
    "--lengths",
    "lengths",
    default="",
    help="Komma-separierte Liste der Highlight-Reel-Laengen in Sekunden, z.B. 60,90,120.",
)
@click.option(
    "--clip-lengths",
    "clip_lengths_raw",
    default="",
    help="Komma-separierte Liste der Kurzclip-Laengen in Sekunden, z.B. 5,10,15.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Zeigt nur die geladene Konfiguration an, ohne irgendetwas zu berechnen.",
)
def main(
    input_path: str,
    config_path: str,
    no_ai: bool,
    lengths: str,
    clip_lengths_raw: str,
    dry_run: bool,
) -> None:
    """Autocut Highlight CLI - erzeugt automatisch Highlight-Reels aus
    langem Reise-/Wohnmobil-/Landschafts-Rohmaterial, lokal und ohne
    Cloud-Zwang."""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        click.echo(f"Fehler: {exc}", err=True)
        raise SystemExit(1) from exc

    logger = setup_logging(config.paths.log_dir)
    set_soft_ram_limit(config.resources.ram_soft_limit_mb, logger)

    env = load_env()

    reel_lengths = _parse_int_list(lengths) or config.output.reel_lengths_sec
    clip_lengths = _parse_int_list(clip_lengths_raw) or config.output.clip_lengths_sec

    input_is_dir = Path(input_path).is_dir()
    logger.info(
        "Starte Autocut Highlight CLI fuer %s: %s",
        "Ordner" if input_is_dir else "Datei",
        input_path,
    )

    _print_summary(logger, input_path, config, no_ai, reel_lengths, clip_lengths, env)

    if dry_run:
        logger.info("Dry-Run beendet (keine Video-Verarbeitung durchgefuehrt).")
        return

    if input_is_dir:
        video_files = sorted(
            p for p in Path(input_path).iterdir() if p.suffix.lower() == ".mp4"
        )
        if not video_files:
            logger.warning("Keine .mp4-Dateien in %s gefunden.", input_path)
            return
        for index, video_path in enumerate(video_files, start=1):
            logger.info("--- Video %d/%d: %s ---", index, len(video_files), video_path.name)
            _run_analysis_for_video(str(video_path), config, logger, reel_lengths, clip_lengths)
    else:
        _run_analysis_for_video(input_path, config, logger, reel_lengths, clip_lengths)


def _format_hms(seconds: float) -> str:
    """Formatiert Sekunden als HH:MM:SS fuer eine lesbare Konsolen-Ausgabe."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _run_analysis_for_video(
    video_path: str,
    config: Config,
    logger: logging.Logger,
    reel_lengths: list[int],
    clip_lengths: list[int],
) -> None:
    """Fuehrt die komplette Pipeline fuer ein einzelnes Video aus:
    Analyse (Schritt 2/3), Score-Fusion/Segmentauswahl (Schritt 4) und
    Video-Export (Schritt 5: Reels + Kurzclips + mehrere Formate)."""
    result = run_analysis(video_path, config, logger)
    warn_if_high_memory(logger, config.resources.ram_warn_mb)

    logger.info("Analyse abgeschlossen fuer: %s", video_path)
    logger.info("  Videolaenge:      %.1f Sekunden", result.duration)
    logger.info("  Proxy:            %s", result.proxy_path)
    logger.info("  Motion-Buckets:   %d Zeitfenster", len(result.motion_buckets))
    logger.info("  Audio-Buckets:    %d Zeitfenster", len(result.audio_buckets))
    logger.info("  Erkannte Beats:   %d Snap-Punkte", len(result.snap_points))
    logger.info("  Stille-Segmente:  %d", sum(1 for s in result.silence_segments if s.get("silent")))

    preview_count = min(5, len(result.motion_buckets))
    if preview_count:
        logger.info("  Erste %d Zeitfenster (Motion / Audio Score):", preview_count)
        for i in range(preview_count):
            m = result.motion_buckets[i]
            a = result.audio_buckets[i] if i < len(result.audio_buckets) else None
            audio_score = f"{a.score:.2f}" if a else "n/a"
            logger.info(
                "    %6.1fs - %6.1fs: motion=%.2f audio=%s",
                m.start, m.end, m.score, audio_score,
            )

    # Score-Fusion (Schritt 4): noch ohne LLM-Scores (folgen in Schritt 7).
    scored_windows = fuse_scores(
        result.motion_buckets, result.audio_buckets, llm_scores=None, weights=config.weights
    )
    selected = select_buckets(scored_windows, result.duration, config.buckets_per_minute)
    logger.info(
        "  Segmentauswahl:   %d von %d Zeitfenstern ausgewaehlt (Buckets/Minute: %.2f)",
        len(selected),
        len(scored_windows),
        config.buckets_per_minute,
    )

    edit_plans_by_length: dict[int, list[tuple[float, float]]] = {}
    for reel_length in reel_lengths:
        edit_plan = build_edit_plan(
            selected, result.snap_points, float(reel_length), config.beats.max_snap_distance_sec
        )
        edit_plans_by_length[reel_length] = edit_plan
        total_selected = sum(end - start for start, end in edit_plan)
        logger.info(
            "  Edit-Plan fuer %ds-Reel: %d Segment(e), %.1fs Gesamtlaenge",
            reel_length,
            len(edit_plan),
            total_selected,
        )
        for start, end in edit_plan:
            logger.info("    %s - %s", _format_hms(start), _format_hms(end))

    # Video-Export (Schritt 5): echte MP4-Reels + Kurzclips erzeugen.
    hw_encoder = detect_hw_encoder(logger)
    cache_dir = video_cache_dir(video_path, config.paths.cache_dir)
    video_name = Path(video_path).stem
    video_output_dir = Path(config.output.output_dir) / video_name

    export_result = export_all(
        edit_plans_by_length,
        video_path,
        hw_encoder,
        cache_dir,
        video_output_dir,
        config.output.formats,
        clip_lengths,
        logger,
    )
    logger.info(
        "  Export abgeschlossen: %d Reel(s), %d Kurzclip(s) in %s",
        len(export_result["reels"]),
        len(export_result["clips"]),
        video_output_dir,
    )
    for path in export_result["reels"]:
        logger.info("    Reel: %s", path)


cli = main


if __name__ == "__main__":
    main()
