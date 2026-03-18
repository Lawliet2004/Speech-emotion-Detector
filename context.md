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

## What Exists In The Repo
- `scripts/create_audio_manifest.py`
- `scripts/audit_dataset.py`
- `scripts/create_project_documentation_pdf.py`
- `scripts/create_project_presentation.py`
- `scripts/create_project_presentation_premium.py`
- `train.py`
- `debug_train.py`
- `skill.md`
- `context.md`
- `reports/data_audit.md`

## Key Decisions
- Keep one raw dataset directory and manage splits through manifests
- Use a controlled desktop V1 before broader real-world claims
- Use the six-class label set for V1
- Keep intensity as metadata only for now
- Exclude exact duplicate audio pairs conservatively during preprocessing

## Current State
- The repo currently contains the core scripts and training entrypoint
- Raw dataset and processed audio are intentionally not included in version control
- A CUDA-enabled PyTorch install is still required in the user environment to run GPU training

## Recommended Next Step
- Install CUDA-enabled PyTorch locally
- Run `train.py`
- Evaluate validation/test metrics
- Then add real-time inference code for live speech prediction

## Maintenance Note
If project decisions, files, training strategy, or deployment assumptions change, update this file in the same work session.
