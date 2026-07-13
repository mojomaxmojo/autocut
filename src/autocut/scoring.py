"""Score-Fusion, Segmentauswahl und Snap-to-Beat (Schritt 4 aus
FEATURE-PLAN.md).

Reine Datenverarbeitung, kein I/O und kein subprocess-Aufruf - alles
hier arbeitet nur mit den Ergebnissen aus analyse.py (Motion-/Audio-
Buckets, Snap-Punkte) und optional spaeter mit LLM-Segment-Scores
(Schritt 7). Ohne LLM-Scores werden die Gewichte fuer motion/audio
automatisch proportional hochskaliert, damit kein Gewicht "verloren"
geht.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from .analyse import Bucket
from .config import Weights


@dataclass
class ScoredWindow:
    """Ein Zeitfenster mit finalem, fusioniertem Score (0.0-1.0)."""

    start: float
    end: float
    score: float


def normalize_weights_without_llm(weights: Weights) -> tuple[float, float]:
    """Skaliert motion/audio-Gewichte proportional hoch, wenn kein
    LLM-Score verfuegbar ist, damit die Summe weiterhin 1.0 ergibt.

    Beispiel: motion=0.2, audio=0.4, llm=0.4 -> ohne llm wird daraus
    motion=0.333..., audio=0.666... (Verhaeltnis 0.2:0.4 bleibt erhalten).
    """
    total = weights.motion + weights.audio
    if total <= 0:
        # Entartungsfall (beide Gewichte 0) - gleich verteilen, damit
        # keine Division durch 0 entsteht.
        return 0.5, 0.5
    return weights.motion / total, weights.audio / total


def _llm_score_for_window(llm_scores: list[dict], start: float, end: float) -> float | None:
    """Findet den durchschnittlichen LLM-Score (0-10, wird hier auf
    0.0-1.0 normalisiert) aller Transkript-Segmente, die sich mit dem
    Zeitfenster [start, end) ueberlappen. Gibt None zurueck, wenn es
    keine Ueberlappung gibt (dann faellt dieses einzelne Fenster auf
    motion+audio zurueck, siehe fuse_scores)."""
    overlapping = [
        s["score"] for s in llm_scores if s["start"] < end and s["end"] > start
    ]
    if not overlapping:
        return None
    avg = sum(overlapping) / len(overlapping)
    return max(0.0, min(1.0, avg / 10.0))


def fuse_scores(
    motion_buckets: list[Bucket],
    audio_buckets: list[Bucket],
    llm_scores: list[dict] | None,
    weights: Weights,
) -> list[ScoredWindow]:
    """Berechnet den finalen Score pro Zeitfenster:

        score = motion*w_motion + audio*w_audio + llm*w_llm

    `llm_scores` darf None oder leer sein (Standardfall in --no-ai oder
    wenn kein API_KEY gesetzt ist) - dann werden nur motion+audio
    genutzt, mit proportional hochskalierten Gewichten. Motion- und
    Audio-Buckets muessen dieselben Zeitfenster-Grenzen haben (das ist
    der Fall, da beide in analyse.py mit demselben bucket_window_sec
    und derselben Videolaenge berechnet werden).
    """
    if len(motion_buckets) != len(audio_buckets):
        # Defensiv: sollte durch analyse.py nie passieren, aber falls
        # doch, nutzen wir die kuerzere Liste statt abzustuerzen.
        n = min(len(motion_buckets), len(audio_buckets))
        motion_buckets = motion_buckets[:n]
        audio_buckets = audio_buckets[:n]

    has_llm = bool(llm_scores)
    w_motion_no_llm, w_audio_no_llm = normalize_weights_without_llm(weights)

    scored: list[ScoredWindow] = []
    for m, a in zip(motion_buckets, audio_buckets):
        llm_value = _llm_score_for_window(llm_scores, m.start, m.end) if has_llm else None

        if llm_value is not None:
            score = (
                m.score * weights.motion
                + a.score * weights.audio
                + llm_value * weights.llm
            )
        else:
            # Kein LLM-Score fuer dieses Fenster (oder gar kein LLM
            # aktiv) - motion/audio proportional hochskaliert nutzen.
            score = m.score * w_motion_no_llm + a.score * w_audio_no_llm

        scored.append(ScoredWindow(start=m.start, end=m.end, score=round(max(0.0, min(1.0, score)), 4)))

    return scored


def select_buckets(
    scored_windows: list[ScoredWindow],
    video_duration: float,
    buckets_per_minute: float,
) -> list[ScoredWindow]:
    """Waehlt die besten Zeitfenster ueber die gesamte Videolaenge
    verteilt aus. Die Anzahl der "Highlight-Buckets" (Regionen des
    Videos, aus denen je ein Top-Fenster gewaehlt wird) ist proportional
    zur Videolaenge, nicht fix.

    Beispiel: 10 Minuten Video, buckets_per_minute=0.5 -> 5 Regionen,
    aus jeder wird das Fenster mit dem hoechsten Score gewaehlt.
    """
    if not scored_windows or video_duration <= 0:
        return []

    minutes = video_duration / 60.0
    n_regions = max(1, round(minutes * buckets_per_minute))
    region_length = video_duration / n_regions

    selected: list[ScoredWindow] = []
    for i in range(n_regions):
        region_start = i * region_length
        region_end = video_duration if i == n_regions - 1 else (i + 1) * region_length
        candidates = [w for w in scored_windows if region_start <= w.start < region_end]
        if not candidates:
            continue
        best = max(candidates, key=lambda w: w.score)
        selected.append(best)

    return selected


def snap_to_nearest(timestamp: float, snap_points: list[float]) -> float:
    """Snapt einen Zeitstempel auf den naechstgelegenen Beat/Pause-Punkt.
    Gibt den unveraenderten Zeitstempel zurueck, wenn keine Snap-Punkte
    vorhanden sind (reine Hilfsfunktion, kein I/O)."""
    if not snap_points:
        return timestamp

    pos = bisect.bisect_left(snap_points, timestamp)
    candidates = []
    if pos < len(snap_points):
        candidates.append(snap_points[pos])
    if pos > 0:
        candidates.append(snap_points[pos - 1])

    return min(candidates, key=lambda p: abs(p - timestamp))


def _merge_overlapping(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Verschmilzt ueberlappende oder direkt aneinander angrenzende
    (start, end)-Paare zu zusammenhaengenden Bereichen."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda seg: seg[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def build_edit_plan(
    selected_windows: list[ScoredWindow],
    snap_points: list[float],
    target_length_sec: float,
) -> list[tuple[float, float]]:
    """Baut aus den ausgewaehlten Highlight-Fenstern eine finale Liste
    von (start, end)-Paaren fuer ein Reel der gewuenschten Ziellaenge,
    mit Schnittkanten, die auf die naechsten Snap-Punkte gezogen sind.

    Vorgehen: Fenster nach Score sortiert aufsammeln, bis die
    Ziellaenge erreicht ist (oder alle Fenster verbraucht sind), dann
    Kanten snappen, ueberlappende/aneinander grenzende Bereiche
    verschmelzen und chronologisch sortieren (fuer ein Reel, das durch
    das Video "wandert", statt nur die Segmente nach Score zu ordnen).
    """
    if not selected_windows or target_length_sec <= 0:
        return []

    by_score = sorted(selected_windows, key=lambda w: w.score, reverse=True)

    chosen: list[ScoredWindow] = []
    total = 0.0
    for window in by_score:
        if total >= target_length_sec:
            break
        chosen.append(window)
        total += window.end - window.start

    snapped_pairs = []
    for window in chosen:
        snapped_start = snap_to_nearest(window.start, snap_points)
        snapped_end = snap_to_nearest(window.end, snap_points)
        if snapped_end <= snapped_start:
            # Snapping hat ein entartetes (zu kurzes/negatives) Segment
            # erzeugt - lieber die urspruenglichen, ungesnappten Grenzen
            # behalten als ein kaputtes Segment zu produzieren.
            snapped_start, snapped_end = window.start, window.end
        snapped_pairs.append((round(snapped_start, 2), round(snapped_end, 2)))

    merged = _merge_overlapping(snapped_pairs)
    return sorted(merged, key=lambda seg: seg[0])
