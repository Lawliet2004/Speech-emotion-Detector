#!/usr/bin/env python3
from __future__ import annotations

# --- Windows stability: must be set BEFORE any torch import ---
import os as _os
if _os.name == "nt":
    _os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    _os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
    _os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
import argparse
import json
import math
import os
import random
import re
import wave
from contextlib import nullcontext
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
except Exception as exc:
    raise ImportError(
        "Missing core training dependencies. Install them first, for example: "
        "pip install numpy pandas scikit-learn"
    ) from exc

# Workaround for known PyTorch 2.5.x Windows access violation crash
# in torch.distributed._composable during import (0xC0000005).
# We don't use distributed training, so block the problematic module.
import sys
import types
if os.name == "nt":
    # Windows stability workarounds: disable TorchDynamo/Triton compiler paths.
    # This project does not use torch.compile, so this avoids known crashes safely.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

    _compile_stub = types.ModuleType("torch._compile")

    def _disable_dynamo(fn=None, recursive=True):
        # No-op decorator used by torch optimizers when compile is not needed.
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    _compile_stub._disable_dynamo = _disable_dynamo
    sys.modules.setdefault("torch._compile", _compile_stub)

    _dist_composable = types.ModuleType("torch.distributed._composable")
    _dist_composable.__path__ = []
    sys.modules["torch.distributed._composable"] = _dist_composable

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:
    raise ImportError(
        "PyTorch is required to run train.py. Install a CUDA-enabled build first, "
        "for example: pip install torch --index-url https://download.pytorch.org/whl/cu121"
    ) from exc


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train a GPU-ready baseline speech emotion recognition model."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "manifests" / "processed" / "processed_audio_manifest.csv",
        help="Path to the processed manifest CSV.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=root / "artifacts" / "baseline_gpu",
        help="Directory where checkpoints and reports will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Use 0 by default for Windows/Jupyter compatibility.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Prefer auto unless you need to force cpu/cuda.",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="best_baseline_model.pt",
        help="Checkpoint filename inside artifacts-dir.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_runtime(device: torch.device, requested_device: str) -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Torch CUDA build: {getattr(torch.version, 'cuda', None)}")
    print(f"Requested device: {requested_device}")
    print(f"Resolved device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif requested_device == "auto":
        print(
            "Warning: CUDA was not selected. This usually means CUDA is unavailable "
            "or a CPU-only PyTorch build is installed."
        )


def make_grad_scaler(use_amp: bool):
    """Create a GradScaler compatible with both old and new PyTorch AMP APIs."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def get_autocast_context(device: torch.device, use_amp: bool):
    """Return an autocast context manager compatible with PyTorch versions."""
    if not use_amp:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=(device.type == "cuda"))


def read_pcm16_mono_wav(path: str, sample_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        actual_sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        compression = wav_file.getcomptype()
        raw_bytes = wav_file.readframes(n_frames)

    if compression != "NONE":
        raise ValueError(f"Unsupported compressed WAV: {compression}")
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, found sample width {sample_width}")
    if actual_sample_rate != sample_rate:
        raise ValueError(
            f"Expected {sample_rate} Hz audio, found {actual_sample_rate} Hz"
        )

    audio = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio / 32768.0


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    if audio.shape[0] >= target_length:
        return audio[:target_length]
    padded = np.zeros(target_length, dtype=np.float32)
    padded[: audio.shape[0]] = audio
    return padded


class EmotionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        sample_rate: int,
        clip_seconds: float,
        augment: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * clip_seconds)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        audio = read_pcm16_mono_wav(row["processed_file_path"], self.sample_rate)
        audio = pad_or_trim(audio, self.target_length)

        if self.augment:
            gain = np.random.uniform(0.9, 1.1)
            noise_scale = np.random.uniform(0.0, 0.003)
            audio = audio * gain
            if noise_scale > 0:
                audio = audio + np.random.normal(
                    0.0, noise_scale, size=audio.shape
                ).astype(np.float32)
            audio = np.clip(audio, -1.0, 1.0)

        waveform = torch.tensor(audio, dtype=torch.float32)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return waveform, label


class LogSpectrogram(nn.Module):
    def __init__(self, n_fft: int = 512, win_length: int = 400, hop_length: int = 160):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        spec = spec.abs().clamp_min(1e-5).log()
        return spec.unsqueeze(1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SERBaselineModel(nn.Module):
    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.frontend = LogSpectrogram()
        self.encoder = nn.Sequential(
            ConvBlock(1, 16, dropout),
            ConvBlock(16, 32, dropout),
            ConvBlock(32, 64, dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.frontend(waveform)
        x = self.encoder(x)
        x = self.pool(x)
        return self.head(x)


class AdamWOptimizer:
    """Minimal AdamW implementation to avoid torch.optim Windows crashes."""

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params = [p for p in params if p.requires_grad]
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.betas = betas
        self.eps = float(eps)
        self.state: dict[int, dict[str, torch.Tensor | int]] = {}
        self.param_groups = [{"lr": self.lr}]

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)
        self.param_groups[0]["lr"] = self.lr

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    def step(self) -> None:
        beta1, beta2 = self.betas
        for p in self.params:
            grad = p.grad
            if grad is None:
                continue

            grad_data = grad.detach()
            if self.weight_decay != 0.0:
                grad_data = grad_data.add(p.detach(), alpha=self.weight_decay)

            state = self.state.get(id(p))
            if state is None:
                state = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                }
                self.state[id(p)] = state

            state["step"] = int(state["step"]) + 1
            step = int(state["step"])
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            exp_avg.mul_(beta1).add_(grad_data, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_data, grad_data, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            step_size = self.lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)

            p.data.addcdiv_(exp_avg, denom, value=-step_size)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamWOptimizer | None,
    scaler,
    device: torch.device,
    use_amp: bool,
) -> dict:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    targets_all: list[int] = []
    preds_all: list[int] = []

    outer_context = nullcontext() if training else torch.no_grad()
    with outer_context:
        for waveforms, targets in loader:
            waveforms = waveforms.to(device, non_blocking=(device.type == "cuda"))
            targets = targets.to(device, non_blocking=(device.type == "cuda"))

            if training:
                optimizer.zero_grad(set_to_none=True)

            autocast_context = get_autocast_context(device=device, use_amp=use_amp)
            with autocast_context:
                logits = model(waveforms)
                loss = criterion(logits, targets)

            if training:
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=1)
            targets_all.extend(targets.detach().cpu().tolist())
            preds_all.extend(preds.detach().cpu().tolist())

    avg_loss = running_loss / len(loader.dataset)
    macro_f1 = f1_score(targets_all, preds_all, average="macro")
    accuracy = float(np.mean(np.array(targets_all) == np.array(preds_all)))
    return {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "targets": targets_all,
        "preds": preds_all,
    }


def build_dataloaders(
    manifest: pd.DataFrame,
    sample_rate: int,
    clip_seconds: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], torch.Tensor]:
    label_names = sorted(manifest["emotion"].unique())
    label_to_index = {label: idx for idx, label in enumerate(label_names)}
    manifest = manifest.copy()
    manifest["label"] = manifest["emotion"].map(label_to_index)

    train_df = manifest.loc[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest.loc[manifest["split"] == "val"].reset_index(drop=True)
    test_df = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)

    train_ds = EmotionDataset(train_df, sample_rate, clip_seconds, augment=True)
    val_ds = EmotionDataset(val_df, sample_rate, clip_seconds, augment=False)
    test_ds = EmotionDataset(test_df, sample_rate, clip_seconds, augment=False)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    class_counts = train_df["label"].value_counts().sort_index().to_numpy()
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return train_loader, val_loader, test_loader, label_names, class_weights


def save_reports(
    artifacts_dir: Path,
    label_names: list[str],
    test_metrics: dict,
    metrics_summary: dict,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with (artifacts_dir / "metrics_summary.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_summary, file, indent=2)
        file.write("\n")

    report_df = pd.DataFrame(
        classification_report(
            test_metrics["targets"],
            test_metrics["preds"],
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_df.to_csv(artifacts_dir / "test_classification_report.csv")

    confusion_df = pd.DataFrame(
        confusion_matrix(test_metrics["targets"], test_metrics["preds"]),
        index=[f"true_{label}" for label in label_names],
        columns=[f"pred_{label}" for label in label_names],
    )
    confusion_df.to_csv(artifacts_dir / "test_confusion_matrix.csv")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and os.name != "nt"

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    seed_everything(args.seed)
    describe_runtime(device, args.device)

    manifest = pd.read_csv(args.manifest)

    def _fix_path(p: str) -> str:
        """Normalize WSL/Windows/relative paths to absolute local paths."""
        p_norm = str(p).replace("\\", "/")
        m = re.match(r"^/?mnt/([a-zA-Z])/(.*)", p_norm)
        if m:
            drive, rest = m.group(1).upper(), m.group(2)
            p_norm = f"{drive}:/{rest}"

        path_obj = Path(p_norm)
        if not path_obj.is_absolute():
            path_obj = project_root / path_obj
        return str(path_obj.resolve())

    manifest["processed_file_path"] = manifest["processed_file_path"].map(_fix_path)
    manifest = manifest.loc[manifest["status"] == "ok"].copy()
    if manifest.empty:
        raise RuntimeError(
            "No usable rows were found in the processed manifest. "
            "Run preprocessing first and confirm status == 'ok'."
        )

    missing_mask = ~manifest["processed_file_path"].map(lambda p: Path(p).exists())
    if missing_mask.any():
        sample_missing = manifest.loc[missing_mask, "processed_file_path"].iloc[0]
        raise FileNotFoundError(
            "Some processed audio files do not exist after path normalization. "
            f"Example missing file: {sample_missing}"
        )

    train_loader, val_loader, test_loader, label_names, class_weights = build_dataloaders(
        manifest=manifest,
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    model = SERBaselineModel(num_classes=len(label_names), dropout=args.dropout).to(
        device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamWOptimizer(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    lr_factor = 0.5
    lr_patience = 2
    lr_bad_epochs = 0
    best_lr_metric = -1.0
    scaler = make_grad_scaler(use_amp=use_amp) if use_amp else None

    best_checkpoint_path = args.artifacts_dir / args.save_name
    history: list[dict] = []
    best_val_f1 = -1.0

    print(f"Training rows: {len(train_loader.dataset)}")
    print(f"Validation rows: {len(val_loader.dataset)}")
    print(f"Test rows: {len(test_loader.dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        if val_metrics["macro_f1"] > best_lr_metric:
            best_lr_metric = val_metrics["macro_f1"]
            lr_bad_epochs = 0
        else:
            lr_bad_epochs += 1
            if lr_bad_epochs > lr_patience:
                optimizer.set_lr(optimizer.lr * lr_factor)
                lr_bad_epochs = 0

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "lr": optimizer.lr,
            }
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_names": label_names,
                    "config": {
                        "sample_rate": args.sample_rate,
                        "clip_seconds": args.clip_seconds,
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "dropout": args.dropout,
                    },
                    "best_val_f1": best_val_f1,
                },
                best_checkpoint_path,
            )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

    history_df = pd.DataFrame(history)
    history_df.to_csv(args.artifacts_dir / "training_history.csv", index=False)

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_metrics = run_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        optimizer=None,
        scaler=scaler,
        device=device,
        use_amp=use_amp,
    )
    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        scaler=scaler,
        device=device,
        use_amp=use_amp,
    )

    metrics_summary = {
        "best_val_f1": checkpoint["best_val_f1"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_accuracy": val_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "device": str(device),
        "checkpoint": str(best_checkpoint_path),
    }
    save_reports(args.artifacts_dir, label_names, test_metrics, metrics_summary)

    print(json.dumps(metrics_summary, indent=2))
    print(f"Saved artifacts to: {args.artifacts_dir}")


if __name__ == "__main__":
    main()
