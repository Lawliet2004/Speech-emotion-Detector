#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "docs" / "speech_emotion_project_documentation.pdf"
SPLIT_SUMMARY_PATH = ROOT / "manifests" / "split_summary.json"
AUDIT_PATH = ROOT / "reports" / "data_audit.json"
PROCESSED_MANIFEST_PATH = ROOT / "manifests" / "processed" / "processed_audio_manifest.csv"


NAVY = "#101828"
SLATE = "#364152"
BLUE = "#3776ff"
TEAL = "#15ad9e"
CORAL = "#ff7555"
GOLD = "#f6bd16"
SOFT_BG = "#f3f7fc"
CARD_BG = "#ffffff"
TEXT = "#404858"
MUTED = "#6b7280"
GRID = "#dbe3ee"


def load_data() -> dict:
    with SPLIT_SUMMARY_PATH.open(encoding="utf-8") as file:
        split_summary = json.load(file)
    with AUDIT_PATH.open(encoding="utf-8") as file:
        audit = json.load(file)
    with PROCESSED_MANIFEST_PATH.open(newline="", encoding="utf-8") as file:
        processed_rows = list(csv.DictReader(file))
    return {
        "split_summary": split_summary,
        "audit": audit,
        "processed_rows": processed_rows,
    }


def make_page(fig):
    fig.patch.set_facecolor(SOFT_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def add_title(ax, title: str, subtitle: str | None = None) -> None:
    ax.text(0.06, 0.93, title, fontsize=22, fontweight="bold", color=NAVY, va="top")
    if subtitle:
        ax.text(0.06, 0.895, subtitle, fontsize=10.5, color=MUTED, va="top")


def add_footer(ax, page_num: int) -> None:
    ax.text(0.94, 0.03, f"Page {page_num}", fontsize=9.5, color=MUTED, ha="right")


def add_card(ax, xywh, title: str, body_lines: list[str], accent: str = BLUE) -> None:
    x, y, w, h = xywh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=GRID,
        facecolor=CARD_BG,
    )
    ax.add_patch(box)
    ax.add_patch(Rectangle((x, y + h - 0.018), w, 0.018, color=accent, ec=accent))
    ax.text(x + 0.018, y + h - 0.04, title, fontsize=12.5, fontweight="bold", color=SLATE, va="top")
    text_y = y + h - 0.08
    for line in body_lines:
        ax.text(x + 0.02, text_y, f"• {line}", fontsize=10.6, color=TEXT, va="top")
        text_y -= 0.048


def add_metric(ax, xywh, label: str, value: str, accent: str) -> None:
    x, y, w, h = xywh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=GRID,
        facecolor=CARD_BG,
    )
    ax.add_patch(box)
    ax.add_patch(Rectangle((x, y), 0.012, h, color=accent, ec=accent))
    ax.text(x + 0.03, y + h - 0.045, label, fontsize=10.5, color=MUTED, va="top")
    ax.text(x + 0.03, y + 0.04, value, fontsize=20, fontweight="bold", color=NAVY, va="bottom")


def page_1(pdf: PdfPages, data: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = make_page(fig)
    ax.add_patch(Rectangle((0, 0.78), 1, 0.22, color=NAVY))
    ax.text(0.06, 0.90, "Speech Emotion Detection System", fontsize=24, fontweight="bold", color="white", va="top")
    ax.text(
        0.06,
        0.84,
        "Project documentation for dataset preparation, preprocessing, training setup, and current implementation status.",
        fontsize=11.5,
        color="#d8e2f0",
        va="top",
    )

    split_summary = data["split_summary"]
    processed_rows = data["processed_rows"]

    add_metric(ax, (0.06, 0.65, 0.26, 0.09), "Raw clips", f"{split_summary['total_clips']:,}", BLUE)
    add_metric(ax, (0.37, 0.65, 0.22, 0.09), "Speakers", f"{split_summary['total_speakers']}", TEAL)
    add_metric(ax, (0.64, 0.65, 0.30, 0.09), "Processed clips", f"{len(processed_rows):,}", CORAL)

    add_card(
        ax,
        (0.06, 0.42, 0.41, 0.18),
        "Project objective",
        [
            "Build a production-minded speech emotion recognition pipeline.",
            "Target the first version for local desktop deployment.",
            "Return emotion plus confidence for detected speech segments.",
        ],
        accent=BLUE,
    )
    add_card(
        ax,
        (0.53, 0.42, 0.41, 0.18),
        "Current V1 scope",
        [
            "Six-class emotion classification.",
            "Dataset-driven baseline before larger model expansion.",
            "Real-time inference planned after baseline training validation.",
        ],
        accent=TEAL,
    )
    add_card(
        ax,
        (0.06, 0.16, 0.88, 0.20),
        "Documentation summary",
        [
            "This document records the work completed so far: dataset organization, leakage-safe split generation, data audit, preprocessing, and GPU-oriented training setup.",
            "It is intended for project documentation, review, and handoff purposes.",
        ],
        accent=CORAL,
    )
    add_footer(ax, 1)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_2(pdf: PdfPages, data: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = make_page(fig)
    add_title(ax, "Dataset Overview", "AudioWAV corpus statistics and label structure")
    split_summary = data["split_summary"]

    emotions = list(split_summary["emotion_counts"].keys())
    counts = list(split_summary["emotion_counts"].values())

    chart_ax = fig.add_axes([0.09, 0.53, 0.82, 0.26])
    chart_ax.set_facecolor(CARD_BG)
    bars = chart_ax.bar([e.title() for e in emotions], counts, color=[BLUE, TEAL, CORAL, GOLD, "#7c89ff", "#5cc3a7"])
    chart_ax.set_title("Emotion Distribution", fontsize=14, color=SLATE, pad=12)
    chart_ax.spines[["top", "right", "left"]].set_visible(False)
    chart_ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    chart_ax.tick_params(axis="x", labelrotation=0, labelsize=10)
    chart_ax.tick_params(axis="y", labelsize=9)
    for bar, count in zip(bars, counts):
        chart_ax.text(bar.get_x() + bar.get_width() / 2, count + 10, str(count), ha="center", va="bottom", fontsize=9, color=TEXT)

    add_card(
        ax,
        (0.06, 0.28, 0.42, 0.16),
        "Audio properties",
        [
            "All files were confirmed as mono, 16 kHz, 16-bit PCM.",
            "This consistency simplifies preprocessing and training.",
            "The dataset shape supports repeatable feature extraction.",
        ],
        accent=BLUE,
    )
    add_card(
        ax,
        (0.52, 0.28, 0.42, 0.16),
        "Labeling policy",
        [
            "V1 uses six classes: angry, disgust, fear, happy, neutral, sad.",
            "Intensity remains metadata only in the current project scope.",
            "This keeps the first model simpler and easier to validate.",
        ],
        accent=TEAL,
    )
    add_card(
        ax,
        (0.06, 0.08, 0.88, 0.14),
        "Dataset takeaway",
        [
            "The dataset is balanced enough for a strong baseline, with neutral slightly smaller than the other emotions.",
            "A reproducible pipeline is more important than manual folder reshuffling, so the project uses manifests instead of moving files.",
        ],
        accent=CORAL,
    )
    add_footer(ax, 2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_3(pdf: PdfPages, data: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = make_page(fig)
    add_title(ax, "Data Organization and Quality Checks", "Split strategy and audit summary")

    split_summary = data["split_summary"]
    audit = data["audit"]

    split_labels = [k.title() for k in split_summary["split_counts"].keys()]
    split_clips = [v["clips"] for v in split_summary["split_counts"].values()]
    split_speakers = [v["speakers"] for v in split_summary["split_counts"].values()]

    chart_ax = fig.add_axes([0.08, 0.53, 0.38, 0.25])
    chart_ax.set_facecolor(CARD_BG)
    bars = chart_ax.bar(split_labels, split_clips, color=[BLUE, TEAL, CORAL])
    chart_ax.set_title("Clips per Split", fontsize=13, color=SLATE, pad=10)
    chart_ax.spines[["top", "right", "left"]].set_visible(False)
    chart_ax.grid(axis="y", color=GRID, linewidth=0.8)
    chart_ax.tick_params(axis="x", labelsize=10)
    chart_ax.tick_params(axis="y", labelsize=9)
    for bar, count in zip(bars, split_clips):
        chart_ax.text(bar.get_x() + bar.get_width() / 2, count + 20, str(count), ha="center", fontsize=9, color=TEXT)

    pie_ax = fig.add_axes([0.54, 0.50, 0.34, 0.30])
    pie_ax.set_facecolor(CARD_BG)
    pie_ax.pie(split_speakers, labels=split_labels, colors=[BLUE, TEAL, CORAL], autopct="%1.0f%%", textprops={"fontsize": 9, "color": TEXT})
    pie_ax.set_title("Speaker Distribution", fontsize=13, color=SLATE)

    add_card(
        ax,
        (0.06, 0.26, 0.42, 0.18),
        "Split design",
        [
            "Train / val / test were built speaker-disjoint.",
            "This prevents speaker leakage and over-optimistic accuracy.",
            f"Train: {split_clips[0]:,} clips | Val: {split_clips[1]:,} | Test: {split_clips[2]:,}.",
        ],
        accent=BLUE,
    )
    add_card(
        ax,
        (0.52, 0.26, 0.42, 0.18),
        "Audit findings",
        [
            "No missing files or unreadable WAV files.",
            "No manifest/header mismatch and no split leakage.",
            f"Main review item: {audit['issues'][0] if audit['issues'] else 'No issues found.'}",
        ],
        accent=CORAL,
    )
    add_card(
        ax,
        (0.06, 0.08, 0.88, 0.12),
        "Quality conclusion",
        [
            "The dataset is healthy enough for baseline training. The only notable issue was a very small number of exact duplicate audio pairs, which were handled conservatively during preprocessing.",
        ],
        accent=TEAL,
    )
    add_footer(ax, 3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_4(pdf: PdfPages, data: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = make_page(fig)
    add_title(ax, "Preprocessing and Training Preparation", "What was built to make model training repeatable")

    processed_rows = data["processed_rows"]

    # pipeline diagram
    steps = [
        ("Raw manifest", BLUE),
        ("Duplicate filter", TEAL),
        ("Audio standardization", CORAL),
        ("Processed audio", GOLD),
        ("Training manifest", BLUE),
    ]
    x = 0.07
    for idx, (label, color) in enumerate(steps):
        w = 0.16
        h = 0.08
        box = FancyBboxPatch((x, 0.72), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", linewidth=0, facecolor=color)
        ax.add_patch(box)
        ax.text(x + w / 2, 0.758, label, ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
        if idx < len(steps) - 1:
            ax.arrow(x + w, 0.758, 0.035, 0.0, width=0.002, head_width=0.015, head_length=0.01, color=MUTED, length_includes_head=True)
        x += 0.19

    add_metric(ax, (0.08, 0.56, 0.25, 0.10), "Processed clips", f"{len(processed_rows):,}", BLUE)
    add_metric(ax, (0.38, 0.56, 0.25, 0.10), "Duplicate files removed", "6", CORAL)
    add_metric(ax, (0.68, 0.56, 0.25, 0.10), "Failed rows", "0", TEAL)

    add_card(
        ax,
        (0.06, 0.30, 0.42, 0.18),
        "Preprocessing implementation",
        [
            "A preprocessing notebook was rewritten to avoid unavailable audio libraries.",
            "The final version uses standard WAV handling plus NumPy and Pandas.",
            "Outputs are written to data/processed_audio and tracked by a processed manifest.",
        ],
        accent=BLUE,
    )
    add_card(
        ax,
        (0.52, 0.30, 0.42, 0.18),
        "Training setup created",
        [
            "A GPU-oriented notebook was created for baseline training.",
            "A CLI training script train.py was also added for repeated runs.",
            "Both use the processed manifest as the source of truth.",
        ],
        accent=TEAL,
    )
    add_card(
        ax,
        (0.06, 0.08, 0.88, 0.14),
        "Prepared output for the next phase",
        [
            "The project is now ready for baseline model training as soon as a CUDA-enabled PyTorch build is available in the local environment.",
        ],
        accent=CORAL,
    )
    add_footer(ax, 4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_5(pdf: PdfPages, data: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = make_page(fig)
    add_title(ax, "Current Status and Next Steps", "Documentation summary and forward plan")

    add_card(
        ax,
        (0.06, 0.63, 0.88, 0.18),
        "Completed work",
        [
            "Dataset inventory and leakage-safe split generation.",
            "Formal data audit with duplicate detection and quality checks.",
            "Working preprocessing pipeline and processed manifest generation.",
            "GPU-ready baseline training code in notebook and CLI form.",
        ],
        accent=BLUE,
    )
    add_card(
        ax,
        (0.06, 0.39, 0.42, 0.18),
        "Current blocker",
        [
            "The sandbox environment used for development does not include PyTorch.",
            "Because of that, the training code was written and validated structurally but not executed here.",
        ],
        accent=CORAL,
    )
    add_card(
        ax,
        (0.52, 0.39, 0.42, 0.18),
        "Immediate next steps",
        [
            "Install CUDA-enabled PyTorch locally.",
            "Run train.py or the GPU notebook.",
            "Review validation/test macro-F1 and the confusion matrix.",
        ],
        accent=TEAL,
    )
    add_card(
        ax,
        (0.06, 0.15, 0.88, 0.16),
        "Longer-term roadmap",
        [
            "After the baseline is trained, the next project phase is real-time inference: segmenting live audio, running the model, and returning emotion plus confidence in a desktop application flow.",
        ],
        accent=GOLD,
    )
    ax.text(0.06, 0.08, "End of project documentation document", fontsize=11, color=MUTED)
    add_footer(ax, 5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = load_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT_PATH) as pdf:
        page_1(pdf, data)
        page_2(pdf, data)
        page_3(pdf, data)
        page_4(pdf, data)
        page_5(pdf, data)
    print(f"Saved documentation PDF to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
