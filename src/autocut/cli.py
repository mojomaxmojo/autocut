"""CLI-Einstiegspunkt fuer das Autocut Highlight Tool.

Schritt 1 (Fundament): laedt Konfiguration + .env, richtet Logging und
RAM-Schutz ein und zeigt eine Zusammenfassung der aktiven Einstellungen.
Die eigentliche Video-Pipeline (Analyse, Scoring, Encoding, optionale
KI-Schritte) wird in den naechsten Schritten laut FEATURE-PLAN.md
angebunden.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from .config import Config, ConfigError, load_config, load_env
from .logging_setup import setup_logging
from .resources import set_soft_ram_limit


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
    logger.info(
        "Hinweis: Dies ist Schritt 1 (Fundament). Die eigentliche "
        "Video-Analyse und der Export folgen in den naechsten "
        "Ausbaustufen laut FEATURE-PLAN.md."
    )


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

    logger.info(
        "Pipeline wuerde jetzt starten mit den obigen Einstellungen. "
        "Die Analyse-/Encoding-Schritte werden in den naechsten "
        "Ausbaustufen ergaenzt (siehe FEATURE-PLAN.md, Schritt 2 ff.)."
    )


cli = main


if __name__ == "__main__":
    main()
