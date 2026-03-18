#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import wave
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "manifests" / "audio_manifest.csv"
REPORTS_DIR = ROOT / "reports"
JSON_REPORT_PATH = REPORTS_DIR / "data_audit.json"
MARKDOWN_REPORT_PATH = REPORTS_DIR / "data_audit.md"

EXPECTED_COLUMNS = [
    "clip_id",
    "file_name",
    "file_path",
    "speaker_id",
    "utterance_id",
    "emotion_code",
    "emotion",
    "intensity_code",
    "intensity",
    "sample_rate",
    "channels",
    "sample_width_bytes",
    "num_frames",
    "duration_sec",
    "split",
]

EXPECTED_EMOTIONS = {"angry", "disgust", "fear", "happy", "neutral", "sad"}
EXPECTED_SPLITS = {"train", "val", "test"}
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH_BYTES = 2


def load_manifest() -> list[dict]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise SystemExit(
                f"Unexpected manifest columns.\nExpected: {EXPECTED_COLUMNS}\nFound:    {reader.fieldnames}"
            )
        return list(reader)


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return sorted_values[low]
    weight = index - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def read_wav_header(path: Path) -> dict:
    with closing(wave.open(str(path), "rb")) as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width_bytes = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()
        duration_sec = num_frames / float(sample_rate)
        return {
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width_bytes,
            "num_frames": num_frames,
            "duration_sec": round(duration_sec, 6),
        }


def build_report(rows: list[dict]) -> dict:
    issues = []
    unreadable_files = []
    missing_files = []
    header_mismatches = []
    duplicate_groups = defaultdict(list)
    path_occurrences = Counter()
    clip_occurrences = Counter()
    split_speakers = defaultdict(set)
    speaker_counts = Counter()
    emotion_counts = Counter()
    split_counts = Counter()
    split_emotion_counts = defaultdict(Counter)
    duration_values = []

    for row in rows:
        file_path = Path(row["file_path"])
        path_occurrences[row["file_path"]] += 1
        clip_occurrences[row["clip_id"]] += 1
        split_speakers[row["split"]].add(row["speaker_id"])
        speaker_counts[row["speaker_id"]] += 1
        emotion_counts[row["emotion"]] += 1
        split_counts[row["split"]] += 1
        split_emotion_counts[row["split"]][row["emotion"]] += 1

        if row["emotion"] not in EXPECTED_EMOTIONS:
            issues.append(f"Unexpected emotion label for {row['clip_id']}: {row['emotion']}")
        if row["split"] not in EXPECTED_SPLITS:
            issues.append(f"Unexpected split for {row['clip_id']}: {row['split']}")

        if not file_path.exists():
            missing_files.append(row["file_path"])
            continue

        try:
            header = read_wav_header(file_path)
        except Exception as exc:  # noqa: BLE001
            unreadable_files.append({"file_path": row["file_path"], "error": str(exc)})
            continue

        for key in ("sample_rate", "channels", "sample_width_bytes", "num_frames"):
            manifest_value = int(row[key])
            if manifest_value != header[key]:
                header_mismatches.append(
                    {
                        "file_path": row["file_path"],
                        "field": key,
                        "manifest": manifest_value,
                        "actual": header[key],
                    }
                )

        manifest_duration = round(float(row["duration_sec"]), 6)
        if abs(manifest_duration - header["duration_sec"]) > 1e-6:
            header_mismatches.append(
                {
                    "file_path": row["file_path"],
                    "field": "duration_sec",
                    "manifest": manifest_duration,
                    "actual": header["duration_sec"],
                }
            )

        duration_values.append(header["duration_sec"])
        duplicate_groups[sha1_file(file_path)].append(row["file_path"])

    overlap_counts = {
        "train_val": len(split_speakers["train"] & split_speakers["val"]),
        "train_test": len(split_speakers["train"] & split_speakers["test"]),
        "val_test": len(split_speakers["val"] & split_speakers["test"]),
    }

    sorted_durations = sorted(duration_values)
    q1 = percentile(sorted_durations, 0.25)
    q3 = percentile(sorted_durations, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    duration_outliers = []
    for row in rows:
        duration = float(row["duration_sec"])
        if duration < lower_bound or duration > upper_bound:
            duration_outliers.append(
                {
                    "clip_id": row["clip_id"],
                    "duration_sec": duration,
                    "split": row["split"],
                }
            )

    exact_duplicate_groups = [
        sorted(paths)
        for paths in duplicate_groups.values()
        if len(paths) > 1
    ]
    exact_duplicate_groups.sort(key=lambda group: (len(group) * -1, group[0]))

    if missing_files:
        issues.append(f"Missing files referenced by manifest: {len(missing_files)}")
    if unreadable_files:
        issues.append(f"Unreadable WAV files: {len(unreadable_files)}")
    if header_mismatches:
        issues.append(f"Manifest/header mismatches: {len(header_mismatches)}")
    if any(overlap_counts.values()):
        issues.append(f"Speaker overlap detected across splits: {overlap_counts}")
    if exact_duplicate_groups:
        issues.append(f"Exact duplicate audio groups found: {len(exact_duplicate_groups)}")

    report = {
        "manifest_path": str(MANIFEST_PATH),
        "summary": {
            "total_rows": len(rows),
            "total_speakers": len(speaker_counts),
            "split_counts": dict(split_counts),
            "emotion_counts": dict(sorted(emotion_counts.items())),
            "speaker_clip_count_range": {
                "min": min(speaker_counts.values()) if speaker_counts else 0,
                "max": max(speaker_counts.values()) if speaker_counts else 0,
            },
            "duration_sec": {
                "min": round(sorted_durations[0], 6) if sorted_durations else None,
                "p25": round(q1, 6) if sorted_durations else None,
                "p50": round(percentile(sorted_durations, 0.50), 6) if sorted_durations else None,
                "p75": round(q3, 6) if sorted_durations else None,
                "p95": round(percentile(sorted_durations, 0.95), 6) if sorted_durations else None,
                "max": round(sorted_durations[-1], 6) if sorted_durations else None,
                "iqr_outlier_bounds": {
                    "lower": round(lower_bound, 6) if sorted_durations else None,
                    "upper": round(upper_bound, 6) if sorted_durations else None,
                },
            },
            "audio_format": {
                "expected_sample_rate": EXPECTED_SAMPLE_RATE,
                "expected_channels": EXPECTED_CHANNELS,
                "expected_sample_width_bytes": EXPECTED_SAMPLE_WIDTH_BYTES,
            },
        },
        "checks": {
            "duplicate_manifest_paths": [path for path, count in path_occurrences.items() if count > 1],
            "duplicate_clip_ids": [clip_id for clip_id, count in clip_occurrences.items() if count > 1],
            "missing_files": missing_files,
            "unreadable_files": unreadable_files,
            "header_mismatches": header_mismatches,
            "split_speaker_overlap": overlap_counts,
            "exact_duplicate_audio_groups": exact_duplicate_groups,
            "duration_outlier_count": len(duration_outliers),
        },
        "breakdown": {
            "split_emotion_counts": {
                split: dict(sorted(counts.items()))
                for split, counts in sorted(split_emotion_counts.items())
            },
            "speaker_clip_counts": dict(sorted(speaker_counts.items(), key=lambda item: int(item[0]))),
        },
        "samples": {
            "duration_outliers_first_20": duration_outliers[:20],
            "duplicate_audio_groups_first_20": exact_duplicate_groups[:20],
        },
        "issues": issues,
        "status": "pass" if not issues else "review",
    }
    return report


def format_markdown(report: dict) -> str:
    summary = report["summary"]
    checks = report["checks"]
    lines = [
        "# Dataset Audit Report",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Summary",
        f"- Total rows: {summary['total_rows']}",
        f"- Total speakers: {summary['total_speakers']}",
        f"- Split counts: train={summary['split_counts'].get('train', 0)}, val={summary['split_counts'].get('val', 0)}, test={summary['split_counts'].get('test', 0)}",
        f"- Emotion counts: {json.dumps(summary['emotion_counts'], sort_keys=True)}",
        f"- Speaker clip count range: {summary['speaker_clip_count_range']['min']} to {summary['speaker_clip_count_range']['max']}",
        f"- Duration seconds: min={summary['duration_sec']['min']}, p50={summary['duration_sec']['p50']}, p95={summary['duration_sec']['p95']}, max={summary['duration_sec']['max']}",
        "",
        "## Audit Checks",
        f"- Missing files: {len(checks['missing_files'])}",
        f"- Unreadable files: {len(checks['unreadable_files'])}",
        f"- Manifest/header mismatches: {len(checks['header_mismatches'])}",
        f"- Speaker overlap: {json.dumps(checks['split_speaker_overlap'], sort_keys=True)}",
        f"- Duplicate manifest paths: {len(checks['duplicate_manifest_paths'])}",
        f"- Duplicate clip IDs: {len(checks['duplicate_clip_ids'])}",
        f"- Exact duplicate audio groups: {len(checks['exact_duplicate_audio_groups'])}",
        f"- Duration outliers: {checks['duration_outlier_count']}",
        "",
        "## Findings",
    ]

    if report["issues"]:
        lines.extend(f"- {issue}" for issue in report["issues"])
    else:
        lines.append("- No blocking issues found.")

    outliers = report["samples"]["duration_outliers_first_20"]
    if outliers:
        lines.extend(
            [
                "",
                "## Duration Outlier Samples",
                *[
                    f"- {item['clip_id']} ({item['split']}): {item['duration_sec']} sec"
                    for item in outliers
                ],
            ]
        )

    duplicates = report["samples"]["duplicate_audio_groups_first_20"]
    if duplicates:
        lines.extend(
            [
                "",
                "## Duplicate Audio Samples",
                *[f"- {group}" for group in duplicates],
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    report = build_report(rows)
    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_REPORT_PATH.write_text(format_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print("status", report["status"])
    print("issues", len(report["issues"]))


if __name__ == "__main__":
    main()
