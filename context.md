# Project Context

Last updated: 2026-03-18

## Project Goal
Build a production-minded speech emotion recognition system that can process speech in real time and return the emotion present in the speech.

## Current Product Direction
- Target platform: local desktop
- Real-time behavior: microphone audio -> speech segment -> emotion prediction
- V1 output: `emotion + confidence`
- V1 label set: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`
- Intensity labels are currently metadata only, not the prediction target

## Dataset Status
- Dataset folder: `AudioWAV/`
- Confirmed dataset shape: 7,441 WAV files
- Confirmed audio format: mono, 16 kHz, 16-bit PCM
- Confirmed speaker count: 91
- Confirmed emotion counts:
  - angry: 1271
  - disgust: 1271
  - fear: 1271
  - happy: 1271
  - neutral: 1087
  - sad: 1270

## What Has Already Been Created
- `scripts/create_audio_manifest.py`
  - Parses dataset filenames
  - Extracts speaker, utterance, emotion, intensity, and WAV metadata
  - Creates speaker-disjoint `train`, `val`, and `test` splits
- `scripts/audit_dataset.py`
  - Audits manifest integrity, WAV readability, header consistency, duplicate audio, duration outliers, and split leakage
- `notebooks/preprocess_audio.ipynb`
  - Loads the raw manifest
  - Excludes the known duplicate-audio files conservatively
  - Standardizes audio to mono 16 kHz PCM with light normalization
  - Writes a processed manifest for downstream model training
- `notebooks/preprocess_audio_clean.ipynb`
  - Cleaner replacement notebook for preprocessing
  - Keeps `processed_file_path` owned by the manifest to avoid merge-column collisions
  - Preferred notebook for future preprocessing runs
- `notebooks/train_baseline_gpu.ipynb`
  - GPU-oriented baseline training notebook
  - Uses the processed manifest as input
  - Trains a compact log-spectrogram CNN with mixed precision on CUDA when available
  - Saves checkpoint, training history, and evaluation reports
- `train.py`
  - Simple CLI training entrypoint for the same GPU-oriented baseline model
  - Reads the processed manifest, trains on CUDA when available, and saves artifacts for later inference
  - Easier to run repeatedly than the notebook
- `scripts/create_project_presentation.py`
  - Generates a 10-slide PowerPoint summarizing the work completed in the project
- `presentations/speech_emotion_project_summary.pptx`
  - Generated 10-slide project summary deck
- `scripts/create_project_presentation_premium.py`
  - Generates a more polished 10-slide presentation with stronger visual styling, charts, and custom assets
- `presentations/speech_emotion_project_summary_premium.pptx`
  - Higher-visual presentation deck for demos or polished project review
- `scripts/create_project_documentation_pdf.py`
  - Generates a formal project documentation PDF
- `docs/speech_emotion_project_documentation.pdf`
  - Documentation-focused PDF artifact for project explanation and handoff
- `manifests/audio_manifest.csv`
  - Full manifest for all clips
- `manifests/train_manifest.csv`
- `manifests/val_manifest.csv`
- `manifests/test_manifest.csv`
- `manifests/split_summary.json`
- `reports/data_audit.json`
- `reports/data_audit.md`

## Current Split Summary
- Train: 5,228 clips / 64 speakers
- Val: 1,066 clips / 13 speakers
- Test: 1,147 clips / 14 speakers
- Verified: no speaker overlap between train, val, and test

## Key Decisions Already Made
- Do not move raw WAV files into train/test/val folders
- Keep one raw dataset directory and manage splits through manifests
- Use a controlled desktop V1 rather than claiming broad real-world robustness
- Start with a dataset-only path before adding external data or pretrained encoders
- Treat the current dataset as mostly clean, but handle the 3 exact duplicate-audio pairs before training experiments are finalized
- Current preprocessing notebook uses a conservative duplicate policy: exclude both members of each exact duplicate pair until manual label review
- The current `preprocess_audio.ipynb` was rewritten to use only standard-library WAV handling plus NumPy/Pandas so it works in the installed environment without `librosa` or `soundfile`
- Baseline training is now expected to use `notebooks/train_baseline_gpu.ipynb` and `manifests/processed/processed_audio_manifest.csv`
- Baseline training can now be run either from `notebooks/train_baseline_gpu.ipynb` or from `train.py`

## Important Files To Read First
- `skill.md`
- `context.md`
- `scripts/create_audio_manifest.py`
- `scripts/audit_dataset.py`
- `notebooks/preprocess_audio.ipynb`
- `notebooks/preprocess_audio_clean.ipynb`
- `notebooks/train_baseline_gpu.ipynb`
- `train.py`
- `scripts/create_project_presentation.py`
- `scripts/create_project_presentation_premium.py`
- `scripts/create_project_documentation_pdf.py`
- `presentations/speech_emotion_project_summary.pptx`
- `presentations/speech_emotion_project_summary_premium.pptx`
- `docs/speech_emotion_project_documentation.pdf`
- `manifests/split_summary.json`
- `manifests/audio_manifest.csv`
- `reports/data_audit.md`

## Recent Changes
- Added `scripts/create_audio_manifest.py`
- Generated manifest files in `manifests/`
- Established speaker-disjoint train/val/test split
- Added `skill.md` and `context.md` for future agent handoff
- Added `scripts/audit_dataset.py`
- Generated `reports/data_audit.json` and `reports/data_audit.md`
- Confirmed no missing files, unreadable WAVs, manifest/header mismatches, or speaker leakage
- Found 3 exact duplicate-audio pairs and 163 duration outliers by IQR rule
- Added `notebooks/preprocess_audio.ipynb` to standardize audio and generate a processed manifest
- Fixed a merge-column collision in `notebooks/preprocess_audio.ipynb` so `processed_file_path` remains available after merging results back into the manifest
- Added `notebooks/preprocess_audio_clean.ipynb` as the preferred preprocessing notebook with simpler, collision-free column handling
- Rewrote `notebooks/preprocess_audio.ipynb` to remove unavailable dependency requirements and verified it successfully processed the dataset
- Generated `data/processed_audio/` with 7,435 processed WAV files
- Generated `manifests/processed/processed_audio_manifest.csv` with 7,435 rows and 0 failed preprocessing rows
- Added `notebooks/train_baseline_gpu.ipynb` for GPU-ready baseline training
- Confirmed the local sandbox does not currently have `torch` installed, so training notebook execution still depends on installing PyTorch in the user's environment
- Added `train.py` as a simple CLI training script for the baseline GPU model
- Hardened `train.py` for GPU use with better runtime diagnostics, a Windows-only PyTorch import workaround, clearer CUDA/CPU reporting, and a safer training-step order
- Added `scripts/create_project_presentation.py` and generated `presentations/speech_emotion_project_summary.pptx`
- Added `scripts/create_project_presentation_premium.py` and generated `presentations/speech_emotion_project_summary_premium.pptx`
- Added `scripts/create_project_documentation_pdf.py` and generated `docs/speech_emotion_project_documentation.pdf`

## Next Recommended Step
Preprocessing is complete. Next:
- install a CUDA-enabled PyTorch build in the user's environment
- run `train.py` or `notebooks/train_baseline_gpu.ipynb`
- use `manifests/processed/processed_audio_manifest.csv` as the training source of truth
- evaluate validation and test metrics before iterating on architecture
- use the generated PowerPoint for project review, demos, or documentation handoff
- prefer the premium presentation for a more polished visual walkthrough
- use the generated PDF for formal project documentation or written handoff

## Maintenance Note For Future Agents
If you change code, files, architecture, labels, splits, training strategy, or deployment assumptions, update this file before finishing the task.
