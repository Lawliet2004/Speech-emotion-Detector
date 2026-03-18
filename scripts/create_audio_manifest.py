#!/usr/bin/env python3
import csv
import json
import re
import wave
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "AudioWAV"
OUTPUT_DIR = ROOT / "manifests"

FILENAME_RE = re.compile(
    r"(?P<speaker_id>\d+)_(?P<utterance_id>[A-Z]{3})_(?P<emotion_code>[A-Z]{3})_(?P<intensity_code>[A-Z]{2})\.wav$",
    re.IGNORECASE,
)

EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

INTENSITY_MAP = {
    "HI": "high",
    "LO": "low",
    "MD": "medium",
    "XX": "unspecified",
}

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


def read_wav_metadata(path: Path) -> dict:
    with closing(wave.open(str(path), "rb")) as wav_file:
        frame_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        duration_sec = frame_count / float(frame_rate)
        return {
            "sample_rate": frame_rate,
            "channels": wav_file.getnchannels(),
            "sample_width_bytes": wav_file.getsampwidth(),
            "num_frames": frame_count,
            "duration_sec": round(duration_sec, 6),
        }


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(DATASET_DIR.glob("*.wav")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue

        parts = match.groupdict()
        wav_metadata = read_wav_metadata(path)
        emotion_code = parts["emotion_code"].upper()
        intensity_code = parts["intensity_code"].upper()

        rows.append(
            {
                "clip_id": path.stem,
                "file_name": path.name,
                "file_path": path.as_posix(),
                "speaker_id": parts["speaker_id"],
                "utterance_id": parts["utterance_id"].upper(),
                "emotion_code": emotion_code,
                "emotion": EMOTION_MAP[emotion_code],
                "intensity_code": intensity_code,
                "intensity": INTENSITY_MAP[intensity_code],
                **wav_metadata,
            }
        )
    return rows


def build_speaker_stats(rows: list[dict]) -> dict[str, dict]:
    stats = defaultdict(lambda: {"clip_count": 0, "emotion_counts": Counter()})
    for row in rows:
        speaker_stats = stats[row["speaker_id"]]
        speaker_stats["clip_count"] += 1
        speaker_stats["emotion_counts"][row["emotion"]] += 1
    return stats


def assign_splits(rows: list[dict]) -> dict[str, str]:
    speaker_stats = build_speaker_stats(rows)
    total_clips = len(rows)
    total_speakers = len(speaker_stats)

    split_targets = {split: total_clips * ratio for split, ratio in SPLIT_RATIOS.items()}

    split_speaker_targets = {
        split: int(total_speakers * ratio) for split, ratio in SPLIT_RATIOS.items()
    }
    assigned_speaker_slots = sum(split_speaker_targets.values())
    remaining_slots = total_speakers - assigned_speaker_slots
    if remaining_slots > 0:
        remainders = sorted(
            (
                (split, (total_speakers * ratio) - split_speaker_targets[split])
                for split, ratio in SPLIT_RATIOS.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        for split, _ in remainders[:remaining_slots]:
            split_speaker_targets[split] += 1

    split_stats = {
        split: {"clip_count": 0, "speakers": []}
        for split in SPLIT_RATIOS
    }

    speaker_order = sorted(
        speaker_stats.items(),
        key=lambda item: (-item[1]["clip_count"], int(item[0])),
    )

    assignments = {}
    for speaker_id, speaker_info in speaker_order:
        available_splits = [
            split
            for split in SPLIT_RATIOS
            if len(split_stats[split]["speakers"]) < split_speaker_targets[split]
        ]
        if not available_splits:
            raise RuntimeError("No split capacity available for remaining speakers.")

        def split_rank(split: str) -> tuple[float, float, int, str]:
            clip_target = split_targets[split]
            clip_filled_ratio = split_stats[split]["clip_count"] / clip_target
            speaker_filled_ratio = (
                len(split_stats[split]["speakers"]) / split_speaker_targets[split]
            )
            return (
                clip_filled_ratio,
                speaker_filled_ratio,
                split_stats[split]["clip_count"],
                split,
            )

        best_split = min(available_splits, key=split_rank)

        assignments[speaker_id] = best_split
        split_stats[best_split]["clip_count"] += speaker_info["clip_count"]
        split_stats[best_split]["speakers"].append(speaker_id)

    return assignments


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
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
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict]) -> dict:
    summary = {
        "total_clips": len(rows),
        "total_speakers": len({row["speaker_id"] for row in rows}),
        "emotion_counts": dict(sorted(Counter(row["emotion"] for row in rows).items())),
        "split_counts": {},
    }

    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        summary["split_counts"][split] = {
            "clips": len(split_rows),
            "speakers": len({row["speaker_id"] for row in split_rows}),
            "emotion_counts": dict(
                sorted(Counter(row["emotion"] for row in split_rows).items())
            ),
        }

    return summary


def main() -> None:
    if not DATASET_DIR.exists():
        raise SystemExit(f"Dataset directory not found: {DATASET_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    assignments = assign_splits(rows)
    for row in rows:
        row["split"] = assignments[row["speaker_id"]]

    rows.sort(key=lambda row: (int(row["speaker_id"]), row["file_name"]))

    write_csv(OUTPUT_DIR / "audio_manifest.csv", rows)
    for split in ("train", "val", "test"):
        write_csv(
            OUTPUT_DIR / f"{split}_manifest.csv",
            [row for row in rows if row["split"] == split],
        )

    summary = build_summary(rows)
    with (OUTPUT_DIR / "split_summary.json").open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
        json_file.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
