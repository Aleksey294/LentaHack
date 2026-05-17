from __future__ import annotations

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8n.pt")
    model.train(
        data="data/yolo_price_tags/data.yaml",
        epochs=50,
        imgsz=960,
        batch=4,
        project="runs",
        name="price_tag_mvp",
        exist_ok=True,
        workers=0,
    )


if __name__ == "__main__":
    main()
