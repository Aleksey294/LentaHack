from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import pandas as pd


ROOT = Path("data")
LABEL_DIR = ROOT / "labels"
VIDEO_DIR = ROOT / "videos"
DATASET = ROOT / "yolo_price_tags"
FRAME_OFFSETS_MS = [-600, -400, -200, 0, 200, 400, 600]


def as_float(value):
    if pd.isna(value):
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def prepare_dirs() -> None:
    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ["train", "val"]:
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)


def get_frame(cap: cv2.VideoCapture, frame_timestamp_ms: int):
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = int(round(max(frame_timestamp_ms, 0) * fps / 1000.0))
    if frame_count > 0:
        frame_index = min(frame_index, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    return frame if ok else None


def main() -> None:
    random.seed(42)
    prepare_dirs()

    total_images = 0
    skipped_videos = []

    for csv_path in sorted(LABEL_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path).rename(columns={"wholesale_level_1_coun": "wholesale_level_1_count"})
        if df.empty:
            continue

        video_name = f"{csv_path.stem}.mp4"
        video_path = VIDEO_DIR / video_name
        if not video_path.exists() or video_path.stat().st_size < 100_000:
            skipped_videos.append(f"{video_name}: missing or too small")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            skipped_videos.append(f"{video_name}: cv2 cannot open")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            skipped_videos.append(f"{video_name}: bad size")
            cap.release()
            continue

        for frame_ts, group in df.groupby("frame_timestamp", dropna=True):
            base_frame_ms = int(as_float(frame_ts) or 0)

            lines = []
            for _, row in group.iterrows():
                x_min = as_float(row.get("x_min"))
                y_min = as_float(row.get("y_min"))
                x_max = as_float(row.get("x_max"))
                y_max = as_float(row.get("y_max"))
                if None in [x_min, y_min, x_max, y_max]:
                    continue
                x_min = max(0.0, min(width - 1.0, x_min))
                x_max = max(0.0, min(width - 1.0, x_max))
                y_min = max(0.0, min(height - 1.0, y_min))
                y_max = max(0.0, min(height - 1.0, y_max))
                box_w = x_max - x_min
                box_h = y_max - y_min
                if box_w <= 2 or box_h <= 2:
                    continue

                x_c = (x_min + x_max) / 2 / width
                y_c = (y_min + y_max) / 2 / height
                w = box_w / width
                h = box_h / height
                lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

            if not lines:
                continue

            for offset_ms in FRAME_OFFSETS_MS:
                frame_ms = max(0, base_frame_ms + offset_ms)
                frame = get_frame(cap, frame_ms)
                if frame is None:
                    continue

                split = "val" if random.random() < 0.2 else "train"
                stem = f"{Path(video_name).stem}_{frame_ms}ms"
                image_path = DATASET / "images" / split / f"{stem}.jpg"
                label_path = DATASET / "labels" / split / f"{stem}.txt"
                cv2.imwrite(str(image_path), frame)
                label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                total_images += 1

        cap.release()

    yaml_path = DATASET / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {DATASET.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: price_tag",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"images: {total_images}")
    print(f"dataset: {DATASET}")
    print(f"yaml: {yaml_path}")
    if skipped_videos:
        print("skipped videos:")
        for item in skipped_videos:
            print(f"- {item}")


if __name__ == "__main__":
    main()
