# FEATURE-PLAN.md — Autocut Highlight CLI

Lokales Python-3-CLI-Tool zur automatischen Highlight-Erstellung aus
Reise-/Wohnmobil-/Landschafts-Rohmaterial. Läuft komplett offline
(`--no-ai`) oder optional mit KI-Transkription (whisper.cpp) +
Cloud-Free-Tier-LLM-Scoring (Groq/OpenRouter).

**Zielsystem:** CachyOS (Arch-basiert), Lenovo ThinkPad T550, Intel
Dual-Core CPU, Intel HD Graphics 5500 (iGPU, kein NVENC), 8–16 GB RAM.

---

## Vorab: Was auf dem Laptop installiert werden muss

Diese Pakete müssen auf dem CachyOS/Arch-System vorhanden sein, **bevor**
du das Tool testest. Das Tool selbst installiert nichts davon automatisch
(ist so gewollt: keine root-Rechte, kein automatisches Systemändern).

### System-Pakete (pacman)

```bash
sudo pacman -S python python-pip ffmpeg aubio auto-editor intel-media-driver libva-utils
```

- `ffmpeg` — Video/Audio-Analyse, Encoding (mit VAAPI/QSV-Support, in
  Arch-Repos bereits mit Hardware-Encoder-Unterstützung gebaut)
- `intel-media-driver` — VAAPI-Treiber für Intel HD Graphics 5500
  (Broadwell-Generation nutzt `intel-media-driver` ODER den älteren
  `libva-intel-driver` — das Tool testet beide automatisch zur Laufzeit)
- `libva-utils` — liefert `vainfo`, mit dem das Tool VAAPI-Fähigkeit
  prüft
- `aubio` — Kommandozeilen-Tool `aubioonset`/`aubiotrack` für
  Beat/Onset-Erkennung
- `auto-editor` — evtl. nicht in offiziellen Repos; alternativ per
  `pipx install auto-editor` oder `pip install --user auto-editor`

Falls `intel-media-driver` bei dir keine VAAPI-Beschleunigung zeigt
(`vainfo` liefert Fehler), probiere stattdessen:

```bash
sudo pacman -S libva-intel-driver
```

(Broadwell/HD 5500 wird je nach Kernel/Mesa-Version von einem der beiden
Treiber unterstützt — das Tool loggt beim Start, welcher Pfad aktiv ist,
und fällt sonst automatisch auf `libx264` zurück.)

### whisper.cpp (nur nötig, wenn du den KI-Modus nutzen willst)

whisper.cpp ist kein Arch-Paket, muss aus dem AUR oder manuell gebaut
werden:

```bash
# Variante A: AUR (falls verfügbar, mit yay/paru)
yay -S whisper.cpp

# Variante B: manueller Build
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make
# Modell laden (klein wegen Dual-Core-CPU):
bash ./models/download-ggml-model.sh small
# oder noch kleiner/schneller:
bash ./models/download-ggml-model.sh base
```

Das Tool erwartet den Pfad zur `whisper-cli`/`main`-Binary und zum
Modell in der `config.yaml` (siehe Schritt 1). Ist whisper.cpp nicht
installiert oder der Pfad falsch, überspringt das Tool die
Transkription automatisch mit einer Log-Warnung — **kein Absturz**.

### Python-Pakete (pip, in einer venv)

```bash
cd autocut-highlight-cli
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` wird in Schritt 1 mitgeliefert und enthält u.a.:
`click`, `pyyaml`, `python-dotenv`, `requests`, `psutil`.
Kein `ffmpeg-python`-Zwang — wir rufen `ffmpeg`/`ffprobe` direkt per
`subprocess` auf, das ist auf schwacher Hardware transparenter und
leichter zu debuggen als eine Wrapper-Bibliothek.

### Optional: `.env` für KI-Scoring

```bash
cp .env.example .env
# dann API_KEY=... und API_PROVIDER=groq (oder openrouter) eintragen
```

Ohne `.env`/`API_KEY` läuft alles normal weiter, nur ohne
LLM-Segment-Scoring (Log-Hinweis, kein Fehler).

---

## Schritt 1 — Fundament: Projektstruktur, Config, Utils (kein ffmpeg-Aufruf)

**Ziel:** Projekt ist installierbar, Config lädt, CLI startet und zeigt
Hilfe/Version an. Noch keine echte Video-Verarbeitung.

**Neue Dateien:**
- `requirements.txt` — Python-Abhängigkeiten (`click`, `pyyaml`,
  `python-dotenv`, `requests`, `psutil`)
- `config.yaml` — Alle Schwellenwerte/Gewichte/Bucket-Anzahl an einem
  Ort:
  - `weights: {motion: 0.2, audio: 0.4, llm: 0.4}`
  - `buckets_per_minute`, `silent_speed: 20`
  - `proxy_resolution: 480`
  - `whisper: {binary_path, model_path, language: de}`
  - `llm: {provider: groq, model: ...}`
  - `output: {formats: [16:9, 9:16, 1:1], reel_lengths: [60,90,120], clip_lengths: [5,10,15]}`
  - `resources: {max_parallel_jobs: 2, ram_warn_mb: 6000}`
- `.env.example` — Vorlage `API_KEY=`, `API_PROVIDER=groq`
- `src/autocut/__init__.py` — leeres Package-Init
- `src/autocut/config.py`
  - Funktion `load_config(path: str = "config.yaml") -> Config`:
    liest YAML, validiert Pflichtfelder, gibt ein `dataclass Config`
    zurück
  - Funktion `load_env() -> dict`: liest `.env` via `python-dotenv`,
    gibt `{"api_key": ..., "provider": ...}` zurück, **niemals** einen
    Fehler wenn `.env` fehlt
- `src/autocut/logging_setup.py`
  - Funktion `setup_logging(log_dir: str) -> logging.Logger`:
    richtet File-Handler (`logs/run_<timestamp>.log`) + Konsolen-Handler
    mit Fortschritts-freundlichem Format ein
- `src/autocut/checkpoint.py`
  - Funktion `checkpoint_exists(path: str) -> bool`
  - Funktion `write_checkpoint(path: str, data: dict) -> None`
  - Funktion `read_checkpoint(path: str) -> dict | None`
  - (reine Dateisystem-Helfer, JSON-basiert, ein Checkpoint pro
    Pipeline-Schritt/Video, z.B. `.autocut_cache/<video_hash>/motion.json`)
- `src/autocut/resources.py`
  - Funktion `set_soft_ram_limit(mb: int) -> None`: setzt Soft-Limit via
    `resource.setrlimit(resource.RLIMIT_AS, ...)`, fängt
    `NotImplementedError`/`ValueError` ab (z.B. falls auf manchen
    Systemen nicht unterstützt) und loggt nur eine Warnung
  - Funktion `warn_if_high_memory(logger) -> None`: nutzt `psutil`, um
    aktuellen Prozess-RAM zu prüfen und bei Überschreitung zu loggen
- `src/autocut/cli.py`
  - `click`-Gruppe `cli()` mit Optionen `--input`, `--config`,
    `--no-ai`, `--lengths`, `--clip-lengths`, `--dry-run`
  - Befehl `main(...)`: lädt Config + `.env`, richtet Logging ein,
    gibt aktuell nur eine Zusammenfassung der geladenen Konfiguration
    aus (noch kein echter Video-Schritt) — Platzhalter-Ausgabe
    `"Pipeline würde starten mit: ..."`
- `run.py` (im Projekt-Root) — dünner Einstiegspunkt:
  ```python
  from src.autocut.cli import cli
  if __name__ == "__main__":
      cli()
  ```

**Angepasste Stellen:** keine (Projekt ist komplett neu, keine
Altlasten aus dem Web-Template).

**Neue Pakete:** `pip install -r requirements.txt` (siehe oben).

**TESTHINWEIS (Terminal):**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py --help
python run.py --input ./irgendein_video.mp4 --no-ai --dry-run
```
Erwartung: Hilfetext erscheint, zweiter Befehl zeigt eine geladene
Konfigurationsübersicht (Gewichte, Bucket-Regel, `no_ai: True`) und
erzeugt eine Log-Datei unter `logs/`. Es wird **kein** Video verändert.

---

## Schritt 2 — Analyse-Grundlagen: Proxy-Encode, Motion-Score, Audio-Energie

**Ziel:** Aus einem Input-Video wird ein Proxy erzeugt und motion/audio
Scores werden pro Zeitfenster berechnet und als Checkpoint-JSON
gespeichert. Läuft komplett ohne KI.

**Neue Dateien:**
- `src/autocut/ffmpeg_utils.py`
  - Funktion `detect_hw_encoder() -> str`: prüft per `ffmpeg -encoders`
    und `vainfo`, ob VAAPI/QSV verfügbar ist, gibt `"vaapi"`, `"qsv"`
    oder `"libx264"` zurück (Ergebnis wird geloggt)
  - Funktion `probe_duration(path: str) -> float`: `ffprobe`-Aufruf,
    Videolänge in Sekunden
  - Funktion `run_ffmpeg(args: list[str], logger) -> subprocess.CompletedProcess`:
    zentraler subprocess-Wrapper mit Logging von stderr bei Fehlern
- `src/autocut/analyse.py`
  - Funktion `make_proxy(input_path, cache_dir, resolution, logger) -> str`:
    erzeugt 480p-Proxy via ffmpeg (`scale=-2:480`), Checkpoint-geprüft
  - Funktion `motion_score(proxy_path, cache_dir, logger) -> list[dict]`:
    nutzt `mpdecimate` + `metadata=print` Filter auf dem Proxy, um
    "verworfene/behalten"-Frames pro Zeitfenster zu zählen → daraus
    einen normalisierten Motion-Score pro Sekunde/Bucket
  - Funktion `audio_energy(input_path, cache_dir, logger) -> list[dict]`:
    extrahiert Audiospur, nutzt `astats`/`loudnorm` Filter, berechnet
    RMS/Peak pro Zeitfenster → normalisierter Energie-Score
  - Funktion `run_analysis(input_path, config, logger) -> AnalysisResult`:
    orchestriert Proxy + Motion + Audio, mit Checkpointing (überspringt
    Neuberechnung, wenn `.autocut_cache/<hash>/motion.json` und
    `audio.json` bereits existieren)
- `src/autocut/parallel.py`
  - Funktion `run_parallel(tasks: list[Callable], max_workers: int) -> list`:
    dünner Wrapper um `concurrent.futures.ProcessPoolExecutor`, begrenzt
    auf `config.resources.max_parallel_jobs` (Default 2)

**Angepasste Stellen:**
- `src/autocut/cli.py`, Funktion `main(...)`: statt der
  Platzhalter-Ausgabe aus Schritt 1 wird jetzt `run_analysis(...)`
  aufgerufen und das Ergebnis als Kurz-Zusammenfassung ausgegeben
  (ca. 10 Zeilen, an der Stelle des bisherigen Platzhalter-Prints).

**Neue Pakete:** keine zusätzlichen Python-Pakete. Systemseitig muss
`ffmpeg` mit `mpdecimate`- und `astats`-Filtern verfügbar sein (Standard
bei Arch-`ffmpeg`-Paket).

**TESTHINWEIS (Terminal):**
```bash
python run.py --input ./videos/test.mp4 --no-ai
```
Erwartung: Konsole zeigt Fortschritt ("Erzeuge Proxy...", "Berechne
Motion-Score...", "Berechne Audio-Energie..."), am Ende eine Tabelle/
Liste mit Buckets und Scores. Prüfen: Unter
`.autocut_cache/<hash>/proxy.mp4`, `motion.json`, `audio.json` liegen
Dateien. Zweiter Lauf desselben Befehls muss **deutlich schneller**
sein (Checkpoint-Hinweis "bereits vorhanden, überspringe" im Log).

---

## Schritt 3 — Beat/Pause-Erkennung, Stille-Grobschnitt

**Ziel:** Erkennung von Onsets/Beats (aubio) mit Fallback auf
Zeitraster, sowie Stille-Erkennung via `auto-editor` zur Vorbereitung
der späteren Schnittpunkte.

**Neue Dateien:**
- `src/autocut/beats.py`
  - Funktion `detect_beats(input_path, cache_dir, logger) -> list[float]`:
    ruft `aubioonset`/`aubiotrack` per subprocess auf, parst
    Zeitstempel
  - Funktion `fallback_grid(duration, interval_sec) -> list[float]`:
    erzeugt gleichmäßiges Zeitraster, wenn `detect_beats` zu wenige/
    keine Onsets liefert (Schwellenwert aus `config.yaml`)
  - Funktion `get_snap_points(input_path, config, cache_dir, logger) -> list[float]`:
    kombiniert beides mit Checkpointing (`beats.json`)
- `src/autocut/silence.py`
  - Funktion `run_auto_editor(input_path, cache_dir, silent_speed, logger) -> str`:
    ruft `auto-editor` per subprocess auf (Parameter `--silent-speed`
    aus Config), gibt Pfad zur editierten/markierten Ausgabe bzw. zur
    erzeugten Schnittliste zurück, Checkpoint-geprüft
  - Funktion `merge_silence_with_snap(snap_points, silence_data) -> list[float]`:
    reine Datenverarbeitung, kein I/O

**Angepasste Stellen:**
- `src/autocut/analyse.py`, Funktion `run_analysis(...)` (aus Schritt 2):
  erweitert um Aufruf von `get_snap_points(...)` und
  `run_auto_editor(...)`, Ergebnis wird dem `AnalysisResult`
  hinzugefügt (Feld `snap_points`, Feld `silence_segments`).
- `src/autocut/cli.py`, Ausgabe-Zusammenfassung: zwei zusätzliche
  Zeilen ("Erkannte Beats: N", "Stille-Segmente: M").

**Neue Pakete:** keine (System: `aubio`, `auto-editor` müssen
installiert sein, siehe Vorab-Abschnitt).

**TESTHINWEIS (Terminal):**
```bash
python run.py --input ./videos/test.mp4 --no-ai
```
Erwartung: Log zeigt "aubio: X Onsets erkannt" ODER "zu wenige Onsets,
nutze Zeitraster-Fallback". Zusätzlich "auto-editor: N Stille-Segmente
erkannt". Bei einem Video **ohne** klare Beats (z.B. reines
Landschafts-Video ohne Musik) muss der Fallback zuverlässig greifen,
ohne Fehler/Absturz.

---

## Schritt 4 — Score-Fusion, Segmentauswahl, Snap-to-Beat (weiterhin ohne KI)

**Ziel:** Aus Motion+Audio (+ optional LLM, aber noch nicht in diesem
Schritt) wird ein finaler Score pro Zeit-Bucket berechnet, die besten
Segmente werden über die gesamte Videolänge verteilt ausgewählt und auf
die nächsten Snap-Points gezogen.

**Neue Dateien:**
- `src/autocut/scoring.py`
  - Funktion `fuse_scores(motion, audio, llm_scores, weights) -> list[dict]`:
    reine Berechnung, `llm_scores` darf `None`/leer sein (dann wird nur
    mit `motion`+`audio` normalisiert und die Gewichte entsprechend neu
    skaliert, siehe unten)
  - Funktion `normalize_weights_without_llm(weights) -> dict`: wenn kein
    LLM-Score vorhanden ist, werden `w1`/`w2` proportional
    hochskaliert, damit die Summe weiterhin 1.0 ergibt (kein
    "verlorenes" Gewicht)
  - Funktion `select_buckets(scored_segments, video_duration, buckets_per_minute) -> list[dict]`:
    berechnet Bucket-Anzahl proportional zur Videolänge, wählt
    Top-Segmente pro Bucket, verteilt über die gesamte Videolänge
  - Funktion `snap_to_nearest(timestamp, snap_points) -> float`: reine
    Hilfsfunktion für Schnittpunkt-Snapping
  - Funktion `build_edit_plan(selected_segments, snap_points, target_length) -> list[tuple[float, float]]`:
    erzeugt finale (start, end)-Paare für eine gewünschte
    Reel-Zielänge, snapped an Beat/Pause-Timestamps

**Angepasste Stellen:**
- `src/autocut/cli.py`, Funktion `main(...)`: nach `run_analysis(...)`
  wird jetzt `fuse_scores(...)` und `select_buckets(...)` aufgerufen,
  Ergebnis (Edit-Plan) wird in der Konsole als Liste der gewählten
  Zeitfenster ausgegeben (noch kein echter Video-Export).
- `config.yaml`: keine Strukturänderung nötig — die in Schritt 1
  angelegten `weights`- und `buckets_per_minute`-Felder werden jetzt
  erstmals tatsächlich genutzt.

**Neue Pakete:** keine.

**TESTHINWEIS (Terminal):**
```bash
python run.py --input ./videos/test.mp4 --no-ai
```
Erwartung: Konsole listet z.B. "Ausgewählte Segmente: 00:03:12–00:03:22,
00:07:45–00:07:58, ..." — die Zeitstempel sollen über die **gesamte**
Videolänge verteilt sein (nicht nur am Anfang) und exakt auf die zuvor
erkannten Beat/Pause-Punkte fallen (mit den Werten aus Schritt 3
vergleichbar).

---

## Schritt 5 — Encoding/Export: Highlight-Reels + Kurzclips + Formate

**Ziel:** Aus dem Edit-Plan werden echte MP4-Dateien erzeugt: Reels in
60/90/120s, daraus automatisch Kurzclips (5/10/15s), sowie die drei
Seitenverhältnisse 16:9/9:16/1:1. Hardware-Encoder wird automatisch
genutzt, Original bleibt unverändert.

**Neue Dateien:**
- `src/autocut/encode.py`
  - Funktion `build_ffconcat(edit_plan, input_path, cache_dir) -> str`:
    erzeugt `ffconcat`-Datei mit **absoluten Pfaden**, plattformrobust
  - Funktion `render_reel(ffconcat_path, output_path, hw_encoder, aspect_ratio, logger) -> str`:
    ein ffmpeg-Aufruf mit passendem Encoder (`vaapi`/`qsv`/`libx264`,
    Ergebnis aus `detect_hw_encoder()`), inkl. `-vf` Crop/Pad je
    Seitenverhältnis, `-force_key_frames` an den Schnittpunkten
  - Funktion `export_reels(edit_plan, config, hw_encoder, output_dir, logger) -> list[str]`:
    erzeugt alle konfigurierten Reel-Längen (`60,90,120` etc.) × alle
    konfigurierten Formate
  - Funktion `split_into_clips(reel_path, clip_lengths, output_dir, logger) -> list[str]`:
    nutzt ffmpeg `segment`-Muxer, erzeugt Ordnerstruktur
    `clips/highlight_60s/5s_000.mp4` usw. für jede konfigurierte
    Kurzclip-Länge

**Angepasste Stellen:**
- `src/autocut/cli.py`, Funktion `main(...)`: nach der Segmentauswahl
  aus Schritt 4 wird jetzt `export_reels(...)` und
  `split_into_clips(...)` aufgerufen; die bisherige reine Text-Ausgabe
  des Edit-Plans bleibt als Log-Zeile erhalten, zusätzlich werden am
  Ende die erzeugten Dateipfade aufgelistet.
- `src/autocut/ffmpeg_utils.py`, Funktion `detect_hw_encoder()` (aus
  Schritt 2): keine Logikänderung, wird hier lediglich zum ersten Mal
  für einen echten Encode-Aufruf konsumiert statt nur geloggt.

**Neue Pakete:** keine.

**TESTHINWEIS (Terminal + Dateisystem):**
```bash
python run.py --input ./videos/test.mp4 --no-ai --lengths 60,90,120 --clip-lengths 5,10,15
```
Erwartung:
- Ordner `output/<videoname>/reels/` enthält z.B.
  `highlight_60s_16x9.mp4`, `highlight_60s_9x16.mp4`,
  `highlight_60s_1x1.mp4` usw. für 60/90/120s
- Ordner `output/<videoname>/clips/highlight_60s/` enthält
  `5s_000.mp4`, `5s_001.mp4`, ... sowie entsprechende `10s_*`/`15s_*`
- Die Original-Datei unter `--input` ist unverändert (Prüfung:
  Dateigröße/Zeitstempel vor/nach dem Lauf vergleichen)
- Im Log steht, welcher Encoder benutzt wurde (`vaapi`, `qsv` oder
  `libx264` als Fallback)
- Videos lassen sich mit einem normalen Player (z.B. `mpv output/.../highlight_60s_16x9.mp4`)
  abspielen

---

## Schritt 6 — Optionaler KI-Layer: whisper.cpp-Transkription

**Ziel:** Wenn `--no-ai` NICHT gesetzt ist, wird das Audio lokal per
whisper.cpp transkribiert (deutsches Modell, klein). Fehlt whisper.cpp
oder das Modell, läuft die Pipeline trotzdem weiter (wie ohne KI) mit
einer klaren Log-Warnung.

**Neue Dateien:**
- `src/autocut/transcribe.py`
  - Funktion `whisper_available(config, logger) -> bool`: prüft, ob
    die konfigurierte Binary + Modell-Datei existieren und ausführbar
    sind
  - Funktion `transcribe(input_path, config, cache_dir, logger) -> list[dict] | None`:
    ruft whisper.cpp per subprocess auf (Sprache `de`, kleines Modell),
    parst die Zeitstempel-Segmente aus der Ausgabe, Checkpoint-geprüft
    (`transcript.json`); gibt `None` zurück (statt Exception), wenn
    `whisper_available()` `False` liefert oder der Aufruf fehlschlägt

**Angepasste Stellen:**
- `src/autocut/cli.py`, Funktion `main(...)`: direkt nach
  `run_analysis(...)` (Schritt 2/3) wird jetzt, **nur wenn `--no-ai`
  nicht gesetzt ist**, `transcribe(...)` aufgerufen; das Ergebnis
  (`None` oder Segment-Liste) wird als neues Feld an den weiteren
  Pipeline-Aufruf aus Schritt 4 durchgereicht (Parameter
  `transcript_segments=None` als Default in `fuse_scores(...)`, sodass
  Schritt 4 unverändert funktioniert, wenn kein Transkript vorliegt).

**Neue Pakete:** keine Python-Pakete. Systemseitig: whisper.cpp-Binary
+ Modell müssen installiert sein (siehe Vorab-Abschnitt), sonst nur
Log-Warnung, kein Fehler.

**TESTHINWEIS (Terminal):**
```bash
# Ohne whisper.cpp installiert:
python run.py --input ./videos/test.mp4
# Erwartung: Log "whisper.cpp nicht gefunden, überspringe Transkription" — Pipeline läuft trotzdem bis zum Export durch (wie in Schritt 5 getestet).

# Mit installiertem whisper.cpp:
python run.py --input ./videos/test.mp4
# Erwartung: Log zeigt Transkriptions-Fortschritt, danach eine Vorschau der ersten paar Segmente mit Zeitstempeln und deutschem Text in der Konsole.
```

---

## Schritt 7 — Optionaler KI-Layer: LLM-Segment-Scoring (Groq/OpenRouter)

**Ziel:** Transkript-Segmente werden per Cloud-Free-Tier-LLM
bewertet (0–10, "Highlight-Würdigkeit"). Ohne `API_KEY` in `.env`
automatisch sauberer Fallback ohne Fehler. Ergebnis fließt in die
Score-Fusion aus Schritt 4 ein.

**Neue Dateien:**
- `src/autocut/llm_scoring.py`
  - Funktion `build_prompt(segment_text: str) -> str`: baut den
    Bewertungs-Prompt (Ankunft, Wetter, Ausrufe, Tiere, Aussicht/
    Sonnenuntergang, wichtige Reise-Infos hoch bewerten; Füllwörter/
    Wiederholungen niedrig)
  - Funktion `score_segment_groq(text, api_key, model) -> float | None`
  - Funktion `score_segment_openrouter(text, api_key, model) -> float | None`
  - Funktion `score_segments(transcript_segments, env, config, logger) -> list[dict] | None`:
    wählt Provider anhand `config.llm.provider`/`.env`, iteriert über
    Segmente (mit kurzer Pause zwischen Requests wegen Free-Tier-
    Rate-Limits), gibt `None` zurück wenn kein `API_KEY` gesetzt ist —
    **mit Log-Hinweis** ("Kein API_KEY gefunden, LLM-Scoring
    übersprungen")

**Angepasste Stellen:**
- `src/autocut/cli.py`, Funktion `main(...)`: nach `transcribe(...)`
  aus Schritt 6 wird jetzt, nur wenn Transkript vorhanden UND `--no-ai`
  nicht gesetzt, `score_segments(...)` aufgerufen; Ergebnis wird als
  `llm_scores` Parameter an `fuse_scores(...)` aus Schritt 4
  übergeben (dort war der Parameter bereits vorgesehen, keine
  Signaturänderung nötig).
- `src/autocut/scoring.py`, Funktion `fuse_scores(...)` (aus Schritt 4):
  keine Codeänderung nötig, da `llm_scores=None`-Fall dort bereits seit
  Schritt 4 behandelt wird — dieser Schritt liefert nur erstmals
  echte Werte statt `None`.

**Neue Pakete:** keine zusätzlichen (`requests` ist bereits seit
Schritt 1 in `requirements.txt`).

**TESTHINWEIS (Terminal):**
```bash
# Ohne .env / API_KEY:
python run.py --input ./videos/test.mp4
# Erwartung: Log "Kein API_KEY gefunden, LLM-Scoring übersprungen" — Pipeline läuft trotzdem komplett durch bis zum Export.

# Mit .env (API_KEY=... gesetzt, Groq oder OpenRouter):
python run.py --input ./videos/test.mp4
# Erwartung: Log zeigt pro Segment einen Score 0-10 mit kurzem Text-Ausschnitt, z.B.
# "8/10 – 'Wow, schau mal den Sonnenuntergang an!'"
# Finale Segmentauswahl (siehe Schritt 4-Test) sollte jetzt sichtbar von den hoch bewerteten Aussagen beeinflusst sein.
```

---

## Schritt 8 — Batch-Verarbeitung (Ordner statt Einzeldatei) + Feinschliff

**Ziel:** `--input` akzeptiert auch einen Ordner mit mehreren MP4s,
verarbeitet sie sequenziell (jede Datei nutzt intern weiterhin die
Parallelität aus Schritt 2 für ihre eigenen Analyse-Schritte), und die
CLI-Ausgabe wird zu einer verständlichen Gesamt-Zusammenfassung
ausgebaut.

**Neue Dateien:** keine neuen Module — reine Erweiterung/Orchestrierung
bestehender Funktionen.

**Angepasste Stellen:**
- `src/autocut/cli.py`, Funktion `main(...)`: die bisherige Logik (ein
  Video verarbeiten) wird in eine neue interne Hilfsfunktion
  `process_single_video(path, config, env, no_ai, logger)`
  ausgelagert (reines Verschieben des bestehenden Codes aus Schritt
  1–7, keine Verhaltensänderung); `main(...)` selbst prüft nun, ob
  `--input` eine Datei oder ein Ordner ist (`os.path.isdir`), sammelt
  bei einem Ordner alle `*.mp4`-Dateien und ruft
  `process_single_video(...)` für jede einzeln auf, mit einer
  Gesamt-Fortschrittsanzeige ("Video 2/5: wohnmobil_tag3.mp4").

**Neue Pakete:** keine.

**TESTHINWEIS (Terminal):**
```bash
python run.py --input ./videos/ --lengths 60,90,120 --clip-lengths 5,10,15
```
Erwartung: Konsole zeigt "Verarbeite Video 1/N: ...", "Video 2/N: ...";
für jedes Video entsteht ein eigener Unterordner unter `output/`
(analog zu Schritt 5), und ein zweiter Lauf über denselben Ordner
überspringt bei jeder Datei die bereits berechneten Checkpoints
(Motion/Audio/Transkript) und ist dadurch deutlich schneller — genau
wie beim Einzeldatei-Test in Schritt 2, jetzt aber für alle Dateien im
Ordner gleichzeitig sichtbar.

---

## Checkliste

- [x] Schritt 1 — Fundament: Config, Logging, Checkpointing, CLI-Grundgerüst
- [x] Schritt 2 — Proxy-Encode, Motion-Score, Audio-Energie (Parallelisierung)
- [x] Schritt 3 — Beat/Pause-Erkennung (aubio + Fallback), Stille-Grobschnitt (auto-editor)
- [x] Schritt 4 — Score-Fusion, Segmentauswahl über gesamte Videolänge, Snap-to-Beat
- [x] Schritt 5 — Encoding/Export: Reels (60/90/120s), Kurzclips (5/10/15s), 3 Seitenverhältnisse, HW-Encoder-Erkennung
- [x] Schritt 6 — Optional: whisper.cpp-Transkription mit sauberem Fallback
- [ ] Schritt 4 — Score-Fusion, Segmentauswahl über gesamte Videolänge, Snap-to-Beat
- [ ] Schritt 5 — Encoding/Export: Reels (60/90/120s), Kurzclips (5/10/15s), 3 Seitenverhältnisse, HW-Encoder-Erkennung
- [ ] Schritt 6 — Optional: whisper.cpp-Transkription mit sauberem Fallback
- [ ] Schritt 7 — Optional: LLM-Segment-Scoring (Groq/OpenRouter) mit sauberem Fallback
- [ ] Schritt 8 — Batch-Verarbeitung ganzer Ordner + Gesamt-Fortschrittsanzeige

Nach jedem abgehakten Schritt ist das Projekt vollständig lauffähig und
kann per `python run.py --input ... --no-ai` (oder ohne `--no-ai`, ab
Schritt 6/7) getestet werden, bevor der nächste Schritt begonnen wird.
