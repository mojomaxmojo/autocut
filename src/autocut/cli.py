"""CLI-Einstiegspunkt fuer das Autocut Highlight Tool.

Laedt Konfiguration + .env, richtet Logging und RAM-Schutz ein und
fuehrt die komplette Pipeline aus: Proxy-Encode, Motion-Score,
Audio-Energie, Beat/Stille-Erkennung (Schritt 2/3), Score-Fusion und
Segmentauswahl mit Snap-to-Beat (Schritt 4), Video-Export als
Highlight-Reels + Kurzclips in mehreren Formaten (Schritt 5), optionale
KI-Schritte Transkription + LLM-Scoring (Schritt 6/7). Akzeptiert als
--input entweder eine einzelne Videodatei oder einen Ordner mit
mehreren .mp4-Dateien, die dann sequenziell mit Gesamt-Fortschritts-
anzeige und Fehlerisolation pro Video verarbeitet werden (Schritt 8).
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
from .llm_scoring import score_segments
from .logging_setup import setup_logging
from .merge import merge_videos
from .resources import set_soft_ram_limit, warn_if_high_memory
from .scoring import build_edit_plan, fuse_scores, select_buckets
from .transcribe import transcribe


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
@click.option(
    "--merge",
    "merge_videos_flag",
    is_flag=True,
    default=False,
    help=(
        "Bei einem Ordner als --input: fuegt zuerst ALLE Videos im Ordner zu einem "
        "durchgehenden Stream zusammen und verarbeitet danach nur dieses eine "
        "zusammengefuegte Video (statt jedes Video einzeln, wie im Standard-Batch-Modus)."
    ),
)
def main(
    input_path: str,
    config_path: str,
    no_ai: bool,
    lengths: str,
    clip_lengths_raw: str,
    dry_run: bool,
    merge_videos_flag: bool,
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
        if merge_videos_flag:
            if len(video_files) == 1:
                logger.info("Nur eine Videodatei gefunden - --merge hat keine Wirkung.")
                process_single_video(str(video_files[0]), config, logger, reel_lengths, clip_lengths, no_ai, env)
            else:
                merged_path = merge_videos(
                    [str(p) for p in video_files], config.paths.cache_dir, logger
                )
                logger.info(
                    "%d Videos zusammengefuegt - verarbeite jetzt den kompletten Stream.",
                    len(video_files),
                )
                output_name = Path(input_path).resolve().name or "merged"
                process_single_video(
                    merged_path, config, logger, reel_lengths, clip_lengths, no_ai, env, output_name
                )
        else:
            _run_batch(video_files, config, logger, reel_lengths, clip_lengths, no_ai, env)
    else:
        process_single_video(input_path, config, logger, reel_lengths, clip_lengths, no_ai, env)


def _run_batch(
    video_files: list[Path],
    config: Config,
    logger: logging.Logger,
    reel_lengths: list[int],
    clip_lengths: list[int],
    no_ai: bool,
    env: dict[str, str | None],
) -> None:
    """Verarbeitet mehrere Videos sequenziell (jedes Video nutzt intern
    weiterhin die Parallelitaet aus Schritt 2 fuer seine eigenen
    Analyse-Schritte). Ein Fehler bei einem Video bricht die
    Batch-Verarbeitung NICHT ab - die restlichen Videos werden trotzdem
    verarbeitet, und am Ende gibt es eine Gesamt-Zusammenfassung."""
    total = len(video_files)
    logger.info("Batch-Verarbeitung gestartet: %d Video(s) in der Warteschlange.", total)

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for index, video_path in enumerate(video_files, start=1):
        logger.info("=" * 60)
        logger.info("Verarbeite Video %d/%d: %s", index, total, video_path.name)
        logger.info("=" * 60)
        try:
            process_single_video(str(video_path), config, logger, reel_lengths, clip_lengths, no_ai, env)
        except Exception as exc:  # noqa: BLE001 - bewusst breit, damit ein
            # einzelnes fehlerhaftes Video die restliche Batch-Verarbeitung
            # nicht abbricht (Kernprinzip: robust auch bei unerwarteten
            # Fehlern in einer einzelnen Datei, z.B. beschaedigtes Video).
            logger.error("Video %d/%d fehlgeschlagen (%s): %s", index, total, video_path.name, exc)
            failed.append((video_path.name, str(exc)))
        else:
            succeeded.append(video_path.name)
        logger.info("Fortschritt: %d/%d Video(s) verarbeitet.", index, total)

    logger.info("=" * 60)
    logger.info("Batch-Verarbeitung abgeschlossen: %d/%d erfolgreich.", len(succeeded), total)
    if failed:
        logger.warning("%d Video(s) fehlgeschlagen:", len(failed))
        for name, error in failed:
            logger.warning("  - %s: %s", name, error)
    logger.info("=" * 60)


def _format_hms(seconds: float) -> str:
    """Formatiert Sekunden als HH:MM:SS fuer eine lesbare Konsolen-Ausgabe."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def process_single_video(
    video_path: str,
    config: Config,
    logger: logging.Logger,
    reel_lengths: list[int],
    clip_lengths: list[int],
    no_ai: bool,
    env: dict[str, str | None],
    output_name: str | None = None,
) -> None:
    """Fuehrt die komplette Pipeline fuer ein einzelnes Video aus:
    Analyse (Schritt 2/3), optionale Transkription + LLM-Scoring
    (Schritt 6/7, nur wenn no_ai False ist), Score-Fusion/Segmentauswahl
    (Schritt 4) und Video-Export (Schritt 5: Reels + Kurzclips + mehrere
    Formate). Wird sowohl fuer Einzeldateien als auch (mehrfach) fuer
    Batch-Verarbeitung eines ganzen Ordners (Schritt 8) aufgerufen.

    output_name ueberschreibt den Output-Ordnernamen (Standard: Dateiname
    ohne Endung) - nuetzlich beim Video-Merge (--merge), wo der
    zusammengefuegte Video-Dateiname sonst technisch (z.B. "merged")
    statt aussagekraeftig waere."""
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

    # Optionale Transkription (Schritt 6) - nur wenn --no-ai NICHT
    # gesetzt ist. Liefert None bei fehlendem whisper.cpp/Modell oder
    # Fehler waehrend des Aufrufs - die Pipeline laeuft dann exakt wie
    # im --no-ai Modus weiter (kein Absturz, nur eine Log-Warnung, die
    # bereits in transcribe() ausgegeben wurde).
    transcript_segments: list[dict] | None = None
    if no_ai:
        logger.info("  Transkription:    uebersprungen (--no-ai gesetzt)")
    else:
        cache_dir_for_transcript = video_cache_dir(video_path, config.paths.cache_dir)
        transcript_segments = transcribe(video_path, config.whisper, cache_dir_for_transcript, logger)
        if transcript_segments:
            logger.info("  Transkription:    %d Segment(e)", len(transcript_segments))
            preview = transcript_segments[: min(3, len(transcript_segments))]
            for seg in preview:
                logger.info(
                    "    %6.1fs - %6.1fs: %s",
                    seg["start"], seg["end"], seg["text"],
                )
        else:
            logger.info("  Transkription:    keine (siehe Log-Warnung oben, falls whisper.cpp fehlt)")

    # Optionales LLM-Segment-Scoring (Schritt 7) - nur wenn --no-ai
    # NICHT gesetzt ist UND ein Transkript vorhanden ist. Liefert None
    # bei fehlendem API_KEY oder wenn alle Requests fehlschlagen - die
    # Pipeline laeuft dann ohne LLM-Score weiter (kein Absturz, nur ein
    # Log-Hinweis, der bereits in score_segments() ausgegeben wurde).
    llm_scores: list[dict] | None = None
    if not no_ai and transcript_segments:
        llm_scores = score_segments(transcript_segments, env, config.llm, logger)

    # Score-Fusion (Schritt 4): motion+audio, optional angereichert um
    # den LLM-Score pro Zeitfenster (llm_scores=None faellt automatisch
    # auf proportional hochskalierte motion+audio Gewichte zurueck).
    scored_windows = fuse_scores(
        result.motion_buckets, result.audio_buckets, llm_scores=llm_scores, weights=config.weights
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
    video_name = output_name or Path(video_path).stem
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
