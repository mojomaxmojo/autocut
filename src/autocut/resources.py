"""RAM-Schutz fuer schwache Hardware (Dual-Core ThinkPad T550, 8-16 GB
RAM). Setzt weiche Grenzen und warnt bei hohem Verbrauch, killt aber
nichts hart - ein abgestuerzter Analyse-Prozess waere schlimmer als
ein bisschen mehr RAM-Nutzung.
"""

from __future__ import annotations

import logging

try:
    import resource  # Nur auf Unix verfuegbar (passt zu CachyOS/Linux)
except ImportError:  # pragma: no cover - z.B. auf Windows
    resource = None  # type: ignore[assignment]

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


def set_soft_ram_limit(mb: int, logger: logging.Logger | None = None) -> None:
    """Setzt ein weiches Limit fuer den virtuellen Speicher des aktuellen
    Prozesses. Schlaegt das auf manchen Systemen fehl (z.B. kein Unix,
    oder Limit wird vom Kernel abgelehnt), wird das nur geloggt - kein
    Absturz des Tools deswegen.
    """
    log = logger or logging.getLogger("autocut")
    if resource is None:
        log.debug("RAM-Soft-Limit uebersprungen (resource-Modul nicht verfuegbar)")
        return
    try:
        limit_bytes = mb * 1024 * 1024
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_hard = hard if hard != resource.RLIM_INFINITY and hard < limit_bytes else hard
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, new_hard))
        log.debug("RAM-Soft-Limit gesetzt: %d MB", mb)
    except (ValueError, OSError) as exc:
        log.warning(
            "Konnte RAM-Soft-Limit nicht setzen (%s) - Tool laeuft trotzdem weiter, "
            "ohne diese Absicherung.",
            exc,
        )


def warn_if_high_memory(logger: logging.Logger | None = None, warn_mb: int = 5000) -> None:
    """Prueft den aktuellen RAM-Verbrauch des Prozesses und loggt eine
    Warnung, wenn er den konfigurierten Schwellenwert ueberschreitet.
    Kein Fehler, keine Aktion - nur Sichtbarkeit fuer den Nutzer.
    """
    log = logger or logging.getLogger("autocut")
    if psutil is None:
        log.debug("Speicher-Check uebersprungen (psutil nicht installiert)")
        return
    try:
        process = psutil.Process()
        rss_mb = process.memory_info().rss / (1024 * 1024)
        if rss_mb > warn_mb:
            log.warning(
                "Hoher Speicherverbrauch erkannt: %.0f MB (Warnschwelle: %d MB). "
                "Falls das System spuerbar langsam wird, andere Programme schliessen "
                "oder max_parallel_jobs in config.yaml reduzieren.",
                rss_mb,
                warn_mb,
            )
        else:
            log.debug("Speicherverbrauch: %.0f MB", rss_mb)
    except Exception as exc:  # pragma: no cover - defensiv, soll nie crashen
        log.debug("Speicher-Check fehlgeschlagen: %s", exc)
