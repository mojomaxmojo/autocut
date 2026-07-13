"""Duenner Helfer fuer begrenzte Parallelverarbeitung unabhaengiger
Analyse-Schritte (Motion, Audio, spaeter Transkription).

Bewusste Design-Entscheidung: Die eigentliche Rechenlast liegt bei den
ffmpeg/aubio/whisper.cpp-Subprozessen, nicht im Python-Code selbst.
Deshalb reicht ein ThreadPoolExecutor voellig aus (kein GIL-Problem, da
die Threads meist auf externe Prozesse warten) - das vermeidet die
Pickling-Komplexitaet von multiprocessing und ist auf einem
Dual-Core-System genauso ressourcenschonend, solange die Anzahl der
gleichzeitigen ffmpeg-Aufrufe ueber max_workers begrenzt wird.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def run_parallel(
    tasks: list[Callable[[], Any]],
    max_workers: int,
    logger: logging.Logger | None = None,
) -> list[Any]:
    """Fuehrt eine Liste parameterloser Callables (z.B. via functools.partial
    vorbereitet) mit begrenzter Parallelitaet aus und gibt die Ergebnisse
    in der urspruenglichen Reihenfolge zurueck.

    Wirft eine Exception weiter, falls eine der Aufgaben fehlschlaegt -
    der aufrufende Code entscheidet ueber Fallback-Verhalten.
    """
    log = logger or logging.getLogger("autocut")
    max_workers = max(1, min(max_workers, len(tasks) or 1))
    log.debug("Starte %d Aufgabe(n) mit max. %d parallelen Workern", len(tasks), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task) for task in tasks]
        return [future.result() for future in futures]
