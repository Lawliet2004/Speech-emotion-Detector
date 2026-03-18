#!/usr/bin/env python3
"""Diagnostic script to isolate crash in train.py"""
import sys

LOG = open("debug_train_log.txt", "w", encoding="utf-8")

def log(msg):
    LOG.write(msg + "\n")
    LOG.flush()
    print(msg, flush=True)

log("Step 1: basic imports")
try:
    import argparse, json, random, wave
    from contextlib import nullcontext
    from pathlib import Path
    log("  OK")
except Exception as e:
    log(f"  FAIL: {e}")
    sys.exit(1)

log("Step 2: numpy / pandas / sklearn")
try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    log("  OK")
except Exception as e:
    log(f"  FAIL: {e}")
    sys.exit(1)

log("Step 3: import torch")
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    log(f"  OK - torch {torch.__version__}, cuda={torch.cuda.is_available()}")
except Exception as e:
    log(f"  FAIL: {e}")
    sys.exit(1)

log("Step 4: read manifest")
try:
    root = Path(__file__).resolve().parent
    mf_path = root / "manifests" / "processed" / "processed_audio_manifest.csv"
    mf = pd.read_csv(mf_path)
    mf["processed_file_path"] = mf["processed_file_path"].map(lambda p: str(Path(p)))
    mf = mf.loc[mf["status"] == "ok"].copy()
    log(f"  OK - {len(mf)} rows")
    log(f"  Columns: {list(mf.columns)}")
    log(f"  Splits: {mf['split'].value_counts().to_dict()}")
except Exception as e:
    log(f"  FAIL: {e}")
    sys.exit(1)

log("Step 5: check first processed file")
try:
    first_path = mf.iloc[0]["processed_file_path"]
    log(f"  Path: {first_path}")
    log(f"  Exists: {Path(first_path).exists()}")
    if not Path(first_path).is_absolute():
        log("  WARNING: path is relative, may break if CWD != repo root")
except Exception as e:
    log(f"  FAIL: {e}")

log("Step 6: read one WAV file")
try:
    with wave.open(str(first_path), "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        nf = wf.getnframes()
        raw = wf.readframes(nf)
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    log(f"  OK - {len(audio)} samples, sr={sr}, channels={channels}")
except Exception as e:
    log(f"  FAIL: {e}")
    sys.exit(1)

log("Step 7: resolve device")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  OK - device={device}")
    if device.type == "cuda":
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    log(f"  FAIL: {e}")
    sys.exit(1)

log("Step 8: create model on device")
try:
    from train import SERBaselineModel
    model = SERBaselineModel(num_classes=6, dropout=0.25).to(device)
    log(f"  OK - model on {device}")
except Exception as e:
    log(f"  FAIL: {e}")
    import traceback
    LOG.write(traceback.format_exc())
    LOG.flush()
    sys.exit(1)

log("Step 9: forward pass with dummy data")
try:
    dummy = torch.randn(2, 80000, device=device)
    with torch.no_grad():
        out = model(dummy)
    log(f"  OK - output shape: {out.shape}")
except Exception as e:
    log(f"  FAIL: {e}")
    import traceback
    LOG.write(traceback.format_exc())
    LOG.flush()
    sys.exit(1)

log("Step 10: GradScaler + autocast forward")
try:
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    autocast_context = (
        torch.cuda.amp.autocast(enabled=True)
        if device.type == "cuda"
        else nullcontext()
    )
    model.train()
    dummy = torch.randn(2, 80000, device=device)
    targets = torch.tensor([0, 1], device=device, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    with autocast_context:
        logits = model(dummy)
        loss = criterion(logits, targets)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    log(f"  OK - loss={loss.item():.4f}")
except Exception as e:
    log(f"  FAIL: {e}")
    import traceback
    LOG.write(traceback.format_exc())
    LOG.flush()
    sys.exit(1)

log("Step 11: build_dataloaders")
try:
    from train import build_dataloaders
    train_loader, val_loader, test_loader, label_names, cw = build_dataloaders(
        manifest=mf,
        sample_rate=16000,
        clip_seconds=5.0,
        batch_size=4,
        num_workers=0,
        device=device,
    )
    log(f"  OK - train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    log(f"  Labels: {label_names}")
except Exception as e:
    log(f"  FAIL: {e}")
    import traceback
    LOG.write(traceback.format_exc())
    LOG.flush()
    sys.exit(1)

log("Step 12: fetch one real batch from train_loader")
try:
    batch_waveforms, batch_labels = next(iter(train_loader))
    log(f"  OK - waveforms={batch_waveforms.shape}, labels={batch_labels.shape}")
except Exception as e:
    log(f"  FAIL: {e}")
    import traceback
    LOG.write(traceback.format_exc())
    LOG.flush()
    sys.exit(1)

log("ALL STEPS PASSED - train.py should work")
LOG.close()
