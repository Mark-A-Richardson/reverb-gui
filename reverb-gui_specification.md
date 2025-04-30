# **reverb‑gui** — Extension Specification

> **Purpose** — Turn the upstream *Reverb* ASR + diarization engine into a self‑contained desktop application by adding a lightweight Qt (PySide6) GUI and transparent FFmpeg pre‑processing, while keeping the core inference pipeline unchanged.

---

## 1 Repository Layout

```
reverb-gui/
├── OLD/                       # Legacy UI & helper scripts from previous project (reference only)
├── reverb_gui/                # New application package
│   ├── __init__.py
│   ├── gui/                   # PySide6 widgets & windows
│   ├── pipeline/              # Thin wrappers around Reverb ASR + diarization + conversion
│   └── resources/             # Icons, qss, etc.
├── pyproject.toml             # Poetry‑managed deps (already present)
├── .env.template             # Copy to .env and fill your Hugging Face token & model path
└── README.md                  # To be rewritten after feature completion
```

*Documentation files in the forked repo are outdated; we will revise them ****after**** the GUI & conversion workflow are complete.*

---

## 2 Objectives & Feature Scope

| # | Feature                            | Details                                                                                |   |                                                       |
| - | ---------------------------------- | -------------------------------------------------------------------------------------- | - | ----------------------------------------------------- |
| 1 | **File import**                    | Drag‑and‑drop or file‑picker; accepts common audio/video.                              |   |                                                       |
| 2 | **Automatic conversion**           | Run packaged FFmpeg → 16 kHz / 16‑bit PCM mono WAV (*transparent to user*).            |   |                                                       |
| 3 | **ASR + Diarization**              | Use upstream Reverb **large** model; auto‑download on first run, then keep CLI parity. |   | Use upstream Reverb **large** model; keep CLI parity. |
| 4 | **Speaker‑interleaved transcript** | Word‑timestamp ↔ diarization alignment; save as `.txt`.                                |   |                                                       |
| 5 | **Minimal settings panel**         | Speaker count (auto/manual), output path, filename.                                    |   |                                                       |
| 6 | **Progress UI**                    | Multi‑segment bar: Conversion → Transcription.                                         |   |                                                       |
| 7 | **Packaging**                      | Poetry virtualenv; one‑click Windows build via PyInstaller.                            |   |                                                       |

*Noise reduction & advanced NER/keywords are out of scope for this milestone.*

---

## 3 Technical Requirements

| Component         | Choice / Version               | Notes                             |
| ----------------- | ------------------------------ | --------------------------------- |
| Python runtime    | **3.10 or 3.11**               | Compatible with PySide6 6.7+      |
| GUI framework     | **PySide6**                    | GPL‑free, Qt 6 widgets            |
| ASR / Diarization | **Reverb** fork (submodule)    | Already present in repo root      |
| Audio conversion  | **FFmpeg** static binary (v6+) | Bundled; invoked via `subprocess` |
| Build system      | **Poetry**                     | Isolated env; lockfile committed  |
| Packaging         | **PyInstaller**                | Generates `reverb-gui.exe`        |

### 3.1 Dependencies (pyproject excerpt)

```toml
[tool.poetry.dependencies]
python           = ">=3.10,<3.14"
PySide6          = "^6.7"
torch            = "==2.2.2"        # Use CUDA build if GPU available
openai-whisper   = "*"              # Legacy dependency; not used in v1 pipeline but kept for parity
typeguard        = "^2.0"
wandb            = "*"
pyyaml           = ">=3.12"
huggingface-hub  = ">=0.13.0"       # For automatic model downloads
python-dotenv    = "^1.1.0"         # Loads .env file
numpy            = "<2"
GitPython        = "*"
"pyannote.audio" = "==3.3.1"
intervaltree     = "==3.1.0"
ffmpeg-python    = "^0.2"
```

Poetry will install CPU‑only Torch by default; document that developers with NVIDIA GPUs should run `pip3 install torch==2.2.2+cu121` **inside** the venv.

### 3.2 Model Asset Management

- Create a root‑level file `.env.template` with the following keys:

```dotenv
# .env.template — copy to .env and fill in
HUGGING_FACE_HUB_TOKEN=
REVERB_MODELS_DIR=./models   # absolute or relative path for cached weights
```

- At application startup, call `from dotenv import load_dotenv` to read `.env`.
- Use `huggingface_hub.snapshot_download` to fetch the Reverb ASR model and the `reverb-diarization-v2` checkpoint, caching them under `REVERB_MODELS_DIR` (default `./models`).
- If weights are already present, skip the download.
- **If weights are missing**, open a modal *“Models required”* dialog that states the total download size and offers **Download / Cancel** buttons.
- When the user clicks **Download**, immediately open a second dialog that shows a determinate progress bar fed by `snapshot_download` callbacks and a textual percentage (e.g., “423 MB of 742 MB”).
- If the user cancels, exit the app gracefully with an explanatory message.
- The downloader logic lives in `reverb_gui.utils.model_downloader`. It fires only when models are absent and drives the two‑step UI (prompt ➜ progress) described above.

---

## 4 Implementation Plan

| Phase                    | Deliverables                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------- |
| **1 Bootstrap**          | Poetry env, launchable empty PySide6 window, FFmpeg presence check.                               |
| **2 Pipeline adapter**   | `pipeline/engine.py` — function `transcribe(path: Path) -> List[AlignedLine]`.                    |
| **3 GUI integration**    | Drag‑drop widget → background `QThreadPool` job invoking engine; progress signals.                |
| **4 Old code migration** | Cherry‑pick reusable widgets/utilities from `OLD/` into the new `gui/` package.                   |
| **5 Packaging**          | PyInstaller spec; include FFmpeg binary via `--add-data`. Create Github Action to build artefact. |
| **6 QA & docs**          | Smoke‑test on 2 sample videos; update README + usage gif.                                         |

---

## 5 Key Classes & Modules

| Path                         | Responsibility                                                                                 |   |
| ---------------------------- | ---------------------------------------------------------------------------------------------- | - |
| `reverb_gui.pipeline.engine` | Wraps FFmpeg conversion & Reverb inference; emits Qt signals for progress.                     |   |
| `reverb_gui.gui.mainwindow`  | Hosts drag‑drop area, settings pane, progress bar; triggers model‑download prompt when needed. |   |
| `reverb_gui.gui.settings`    | Loads / saves JSON config (`%APPDATA%/reverb-gui/config.json`).                                |   |
| `reverb_gui.utils.ffmpeg`    | Static helpers to call packaged FFmpeg, parse `-progress` output.                              |   |
| `reverb_gui.utils.aligner`   | Word ↔ speaker alignment logic (see previous canvas guide).                                    |   |

---

## 6 Briefing – Current Repo Status

*(as of commit **``**)*

- The repository is a **fresh fork** of RevAI/reverb with **no GUI** code in main packages.
- Directory `OLD/` contains legacy PySide6 UI files and helper scripts from the abandoned AudioVerba project—use only for reference.
- `pyproject.toml` already lists core dependencies (Torch, pyannote, PySide6, ffmpeg‑python). Poetry installs successfully.
- Upstream documentation files are outdated; we will rewrite after the new GUI workflow is functional.
- Primary tasks: integrate FFmpeg conversion, build new `reverb_gui` package, wire progress signals, and package with PyInstaller.

> **Next action:** start Phase 1 — create `reverb_gui/` skeleton, verify Poetry run, and display a blank main window titled *reverb‑gui*.

