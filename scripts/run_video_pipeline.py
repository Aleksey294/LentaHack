from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
yolo_config_dir = ROOT / ".yolo_config"
yolo_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))

from lenta_cv.pipeline import PriceTagPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hierarchical price tag pipeline on a video.")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--weights", default="runs/detect/runs/price_tag_mvp/weights/best.pt", help="YOLO weights")
    parser.add_argument("--out", default="submission_inference.csv", help="Output CSV path")
    parser.add_argument("--crop-dir", default="runs/pipeline/crops", help="Directory to save tag crops")
    parser.add_argument("--conf", type=float, default=0.20, help="Detection confidence threshold")
    parser.add_argument("--stride", type=int, default=3, help="Process every Nth frame")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = PriceTagPipeline(
        detector_weights=args.weights,
        detector_conf=args.conf,
        frame_stride=max(1, args.stride),
    )
    df = pipeline.process_video(video_path=args.video, crop_dir=args.crop_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"saved {out_path} with {len(df)} rows")


if __name__ == "__main__":
    main()
