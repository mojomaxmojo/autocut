# Autocut Highlight CLI

Lokales Python-3-CLI-Tool zur automatischen Highlight-Erstellung aus
langem Reise-/Wohnmobil-/Landschafts-Rohmaterial. Funktioniert komplett
offline (`--no-ai`); ein optionaler KI-Layer (lokale Transkription per
whisper.cpp + Cloud-Free-Tier-LLM-Scoring) kann zusaetzlich aktiviert
werden, wenn ein API-Key vorhanden ist.

Der aktuelle Stand entspricht **Schritt 7** aus `FEATURE-PLAN.md` -
damit ist die komplette Kern-Pipeline inklusive optionalem KI-Layer
fertig: Fundament + Proxy-Encode/Motion/Audio + Beat-Erkennung (aubio
mit Zeitraster-Fallback) + Stille-Grobschnitt (auto-editor) +
Score-Fusion und Segmentauswahl mit Snap-to-Beat + echter Video-Export
(Reels, Kurzclips, mehrere Seitenverhaeltnisse) + optionale lokale
Transkription via whisper.cpp + optionales LLM-Segment-Scoring
(Groq/OpenRouter). Schritt 8 (Batch-Feinschliff) ist der letzte
verbleibende Ausbauschritt.

Zielsystem: CachyOS (Arch-basiert), Lenovo ThinkPad T550, Intel
Dual-Core CPU, Intel HD Graphics 5500 (iGPU, kein NVENC), 8-16 GB RAM.

## Installation

### 1. System-Pakete (pacman)

```bash
sudo pacman -S python python-pip ffmpeg aubio auto-editor intel-media-driver libva-utils
```

Falls `vainfo` (aus `libva-utils`) keine VAAPI-Beschleunigung fuer die
HD Graphics 5500 anzeigt, alternativ:

```bash
sudo pacman -S libva-intel-driver
```

`auto-editor` ist evtl. nicht in den offiziellen Repos; alternativ:

```bash
pipx install auto-editor
# oder
pip install --user auto-editor
```

### 2. Optional: whisper.cpp (nur fuer den KI-Modus)

**Wichtig:** Im AUR gibt es aktuell nur drei whisper.cpp-Varianten
(`whisper.cpp-cuda`, `whisper.cpp-cuda-bin`, `whisper.cpp-openvino`) -
alle drei setzen entweder eine NVIDIA-GPU (CUDA) voraus oder ziehen das
schwergewichtige OpenVINO-Laufzeitpaket mit, dessen GPU-Beschleunigung
auf der Broadwell-iGPU (HD Graphics 5500) ohnehin meist nicht greift.
**Keine dieser AUR-Pakete auswaehlen** - stattdessen manuell bauen
(reiner CPU-Pfad, schlank, keine unnoetigen Abhaengigkeiten):

```bash
# Falls noch nicht vorhanden: Build-Werkzeuge installieren
sudo pacman -S cmake base-devel

git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make
bash ./models/download-ggml-model.sh small
```

Nach dem Build liegt die Binary unter `whisper.cpp/build/bin/` (Name je
Version z.B. `whisper-cli` oder `main`) und das Modell unter
`whisper.cpp/models/ggml-small.bin` - das entspricht bereits den
Default-Pfaden in `config.yaml`. Falls bei dir andere Pfade entstehen,
in `config.yaml` unter `whisper:` anpassen.

Ist whisper.cpp nicht vorhanden oder der Pfad falsch, laeuft das Tool
trotzdem weiter - nur ohne Transkription (Log-Warnung, kein Fehler).

### 3. Python-Umgebung

```bash
cd autocut-highlight-cli
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Optional: .env fuer LLM-Scoring

```bash
cp .env.example .env
# API_KEY=... und API_PROVIDER=groq (oder openrouter) eintragen
```

Ohne `.env`/`API_KEY` laeuft alles normal weiter, nur ohne
LLM-Segment-Scoring.

## Benutzung

```bash
# Komplett ohne KI (nur Motion-Score + Audio-Energie + Beat/Pause-Erkennung):
python run.py --input ./videos/wohnmobil_tag3.mp4 --lengths 60,90,120 --clip-lengths 5,10,15 --no-ai

# Mit KI-Scoring (wenn API_KEY in .env gesetzt ist):
python run.py --input ./videos/ --lengths 60,90,120 --clip-lengths 5,10,15

# Nur Konfiguration pruefen, ohne etwas zu berechnen:
python run.py --input ./videos/test.mp4 --no-ai --dry-run

# Hilfe anzeigen:
python run.py --help
```

**Aktueller Funktionsumfang (Schritt 2/3):** Fuer jede Eingabedatei wird
ein 480p-Proxy erzeugt (Original bleibt unveraendert), anschliessend
werden parallel berechnet: Motion-Score (Bewegungsintensitaet),
Audio-Energie (Lautstaerke), Beat/Onset-Erkennung (aubio, mit
automatischem Zeitraster-Fallback bei fehlender Musik) und
Stille-Erkennung (auto-editor). Alle Ergebnisse werden in der Konsole
als Vorschau ausgegeben und unter `.autocut_cache/` gecacht - ein
zweiter Lauf mit denselben Dateien ist dadurch deutlich schneller.

Fehlen `aubio` oder `auto-editor` auf dem System, laeuft die Pipeline
trotzdem vollstaendig durch - es wird nur eine Log-Warnung ausgegeben
und automatisch auf ein gleichmaessiges Zeitraster bzw. keine
Stille-Information zurueckgefallen (kein Absturz).

Fuer jede konfigurierte Reel-Laenge (`--lengths`, z.B. 60/90/120s) wird
ein Edit-Plan berechnet: die Zeitfenster mit dem hoechsten fusionierten
Score (motion+audio, proportional hochskaliert ohne KI) werden ueber
die gesamte Videolaenge verteilt ausgewaehlt und auf die naechsten
Beat/Pause-Snap-Punkte gezogen (mit einer maximalen Snap-Distanz, damit
Segmente nicht auf 0 Sekunden kollabieren). Daraus werden anschliessend
echte MP4-Highlight-Reels erzeugt - fuer jedes konfigurierte
Seitenverhaeltnis (`16:9`/`9:16`/`1:1`) eine eigene Datei - sowie
automatisch abgeleitete Kurzclips (`--clip-lengths`, z.B. 5/10/15s) aus
dem jeweils ersten konfigurierten Format. Der genutzte Hardware-Encoder
(VAAPI/QSV/libx264) wird automatisch erkannt und geloggt; schlaegt ein
Hardware-Encoder fehl, wird automatisch auf libx264 (Software)
zurueckgefallen.

Ausgabestruktur (Standard-Ordner `output/`):
```
output/<videoname>/reels/highlight_60s_16x9.mp4
output/<videoname>/reels/highlight_60s_9x16.mp4
output/<videoname>/reels/highlight_60s_1x1.mp4
output/<videoname>/clips/highlight_60s/5s_000.mp4
output/<videoname>/clips/highlight_60s/10s_000.mp4
...
```

**Hinweis zu VAAPI:** Der VAAPI-Encode-Pfad nutzt standardmaessig das
Geraet `/dev/dri/renderD128`. Hat dein System mehrere GPUs/Render-Nodes
(z.B. `renderD129`), oder schlaegt VAAPI-Encoding fehl, faellt das Tool
automatisch auf `libx264` (Software) zurueck - du verlierst dadurch
keine Funktionalitaet, nur etwas Geschwindigkeit. Mit `vainfo` kannst
du pruefen, welches Geraet auf deinem System die HD Graphics 5500 ist.
Bekanntes, harmloses Verhalten auf aelteren Intel-iGPUs (Broadwell):
ffmpeg kann NACH erfolgreichem Encodieren beim Aufraeumen der
VAAPI-Ressourcen abstuerzen (`free(): invalid pointer`) - das Tool
erkennt diesen Fall und wertet die bereits vollstaendig geschriebene
Ausgabedatei trotzdem als Erfolg (kein unnoetiger Re-Encode).

**Optionale Transkription (Schritt 6):** Ohne `--no-ai` wird das Audio
jeder Datei lokal per whisper.cpp transkribiert (deutsches Modell,
Zeitstempel pro Segment). Fehlt die whisper.cpp-Binary oder das
Modell (siehe Installationsabschnitt oben), wird die Transkription
automatisch uebersprungen - kein Fehler, nur eine Log-Warnung, die
Pipeline laeuft normal bis zum Export durch.

**Hinweis zu Halluzinationen/Wiederholungsschleifen:** Bei Audio ohne
klare Sprache (Wind, Motorengeraeusche, Fahrradfahren) kann whisper.cpp
in eine Schleife geraten und denselben (meist falschen) Satz viele Male
identisch wiederholen. Das Tool setzt standardmaessig `--max-context 0`
(keine Kontextuebernahme zwischen Segmenten) und einen erhoehten
`--entropy-thold 2.6`, um das zu reduzieren (beide in `config.yaml`
unter `whisper:` einstellbar). Fuer eine noch robustere Loesung kannst
du zusaetzlich VAD (Voice Activity Detection) aktivieren, die
Nicht-Sprache-Abschnitte bereits vor der Transkription herausfiltert:

```bash
cd whisper.cpp
bash ./models/download-vad-model.sh silero-v6.2.0
```

Danach in `config.yaml` unter `whisper:` eintragen:
```yaml
vad_model_path: "whisper.cpp/models/ggml-silero-v6.2.0.bin"
```

Ohne `vad_model_path` (Standard: leer) laeuft die Transkription normal
weiter, nur ohne VAD-Vorfilterung.

**Optionales LLM-Segment-Scoring (Schritt 7):** Ist ein Transkript
vorhanden UND ein `API_KEY` in `.env` gesetzt (Groq oder OpenRouter,
siehe `.env.example`), wird jedes Transkript-Segment per Cloud-
Free-Tier-LLM auf einer Skala 0-10 bewertet, wie "highlight-wuerdig"
es fuer ein Reise-/Naturvideo ist (Ankunft, Wetter, Ausrufe, Tiere,
Aussicht/Sonnenuntergang hoch; Fuellwoerter/Wiederholungen niedrig).
Das Ergebnis fliesst in die Score-Fusion ein: `weights.llm` aus
`config.yaml` bestimmt, wie stark der LLM-Score gegenueber
motion/audio gewichtet wird. Fehlt der API_KEY, oder schlaegt ein
Request fehl, wird automatisch auf motion+audio zurueckgefallen -
kein Fehler, nur ein Log-Hinweis.

**Hinweis zu OpenRouter-Modellen:** OpenRouters Angebot an `:free`-
Modellen wechselt haeufig - ein heute kostenloses Modell kann morgen
mit `HTTP 404 "unavailable for free"` fehlschlagen. Standardmaessig ist
`llm.openrouter_model` in `config.yaml` auf `openrouter/free` gesetzt
(OpenRouters eigener Free-Router, waehlt automatisch ein aktuell
verfuegbares kostenloses Modell). Willst du ein bestimmtes Modell
erzwingen, pruefe zuerst https://openrouter.ai/models?q=free auf den
aktuellen Modell-Slug. Bei einem 404 laeuft die Pipeline trotzdem
weiter (motion+audio ohne LLM-Score), nur eine Log-Warnung erscheint.

**Hinweis zu auto-editor-Versionen:** Die auto-editor-CLI hat ihre
Export-Flags mehrfach geaendert (aeltere Anleitungen nennen z.B.
`--export json`, was in aktuellen Versionen mit `Unknown export format`
fehlschlaegt). Dieses Tool nutzt das stabile `--export v1`-Format
(https://auto-editor.com/docs/v1). Falls du eine sehr alte oder sehr
neue auto-editor-Version installiert hast und trotzdem einen
`Unknown export format`-Fehler siehst, pruefe mit
`auto-editor --help | grep -A5 -- --export`, welche Formate deine
Version unterstuetzt.

## Konfiguration

Alle Schwellenwerte, Gewichte, Bucket-Anzahl, Modellauswahl etc. liegen
in `config.yaml` - siehe Kommentare in der Datei. Keine Hardcoded-Werte
im Code.

## Projektstruktur

```
run.py                      # Einstiegspunkt
config.yaml                 # Alle Einstellungen
.env.example                # Vorlage fuer API-Keys (LLM-Scoring)
requirements.txt            # Python-Abhaengigkeiten
FEATURE-PLAN.md             # Schrittweiser Ausbauplan
src/autocut/
  cli.py                    # CLI (click), Orchestrierung
  config.py                 # Config- und .env-Laden
  logging_setup.py          # Logging (Konsole + Datei)
  checkpoint.py             # Checkpointing-Helfer
  resources.py              # RAM-Schutz (Soft-Limits, Warnungen)
  ffmpeg_utils.py           # ffmpeg/ffprobe-Helfer, HW-Encoder-Erkennung
  analyse.py                # Proxy-Encode, Motion-Score, Audio-Energie
  parallel.py                # Begrenzte Parallelverarbeitung
  beats.py                  # Beat/Onset-Erkennung (aubio) + Fallback
  silence.py                 # Stille-Erkennung (auto-editor)
  scoring.py                 # Score-Fusion, Segmentauswahl, Snap-to-Beat
  encode.py                  # Video-Export: Reels, Kurzclips, Formate
  transcribe.py               # Optionale Transkription via whisper.cpp
  llm_scoring.py              # Optionales LLM-Segment-Scoring (Groq/OpenRouter)
```

## Ausbauplan

Siehe `FEATURE-PLAN.md` fuer die vollstaendige, schrittweise Roadmap
(Schritt 2: Motion/Audio-Analyse, Schritt 3: Beat/Stille-Erkennung,
Schritt 4: Score-Fusion, Schritt 5: Encoding/Export, Schritt 6+7:
optionaler KI-Layer, Schritt 8: Batch-Verarbeitung).
