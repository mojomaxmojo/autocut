# Autocut Highlight CLI

Lokales Python-3-CLI-Tool zur automatischen Highlight-Erstellung aus
langem Reise-/Wohnmobil-/Landschafts-Rohmaterial. Funktioniert komplett
offline (`--no-ai`); ein optionaler KI-Layer (lokale Transkription per
whisper.cpp + Cloud-Free-Tier-LLM-Scoring) kann zusaetzlich aktiviert
werden, wenn ein API-Key vorhanden ist.

Der aktuelle Stand entspricht **Schritt 1** aus `FEATURE-PLAN.md`
(Fundament: Config, Logging, Checkpointing, RAM-Schutz, CLI-Grundgerueut).
Die eigentliche Video-Analyse/Encoding folgt in den naechsten Schritten.

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
  # weitere Module folgen laut FEATURE-PLAN.md:
  # ffmpeg_utils.py, analyse.py, parallel.py, beats.py, silence.py,
  # scoring.py, encode.py, transcribe.py, llm_scoring.py
```

## Ausbauplan

Siehe `FEATURE-PLAN.md` fuer die vollstaendige, schrittweise Roadmap
(Schritt 2: Motion/Audio-Analyse, Schritt 3: Beat/Stille-Erkennung,
Schritt 4: Score-Fusion, Schritt 5: Encoding/Export, Schritt 6+7:
optionaler KI-Layer, Schritt 8: Batch-Verarbeitung).
