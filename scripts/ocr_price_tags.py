from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import cv2
import easyocr
import pandas as pd
from ultralytics import YOLO


PRICE_RE = re.compile(r"\b\d{1,5}\s*['`’.,]\s*\d{2}\b|\b\d{2,5}\b")
BARCODE_RE = re.compile(r"\b\d{12,14}\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/yolo_price_tags/images/val", help="Image, directory, or video")
    parser.add_argument("--weights", default="runs/detect/runs/price_tag_mvp/weights/best.pt")
    parser.add_argument("--out", default="ocr_predictions.csv")
    parser.add_argument("--conf", type=float, default=0.15)
    return parser.parse_args()


def iter_images(source: Path):
    if source.is_dir():
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            yield from sorted(source.glob(pattern))
    else:
        yield source


def normalize_ocr_result(result) -> list[str]:
    texts: list[str] = []
    for item in result:
        if len(item) >= 2 and str(item[1]).strip():
            texts.append(str(item[1]).strip())
    return texts


def ocr_score(result) -> float:
    if not result:
        return 0.0
    scores = []
    for item in result:
        if len(item) >= 3:
            try:
                scores.append(float(item[2]))
            except (TypeError, ValueError):
                pass
    return sum(scores) / len(scores) if scores else 0.0


def rotate_image(image, angle: int):
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(angle)


def preprocess_for_ocr(crop):
    h, w = crop.shape[:2]
    scale = max(2, min(5, int(900 / max(h, w)) + 1))
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    return gray


def read_best_rotation(reader, crop):
    best = {"angle": 0, "text": "", "score": -1.0, "price": "", "raw": []}
    for angle in [0, 90, 180, 270]:
        rotated = rotate_image(crop, angle)
        prepared = preprocess_for_ocr(rotated)
        result = reader.readtext(prepared, detail=1, paragraph=False)
        texts = normalize_ocr_result(result)
        full_text = " ".join(texts)
        price = extract_price(full_text)
        score = ocr_score(result) + (1.0 if price else 0.0) + min(len(full_text), 80) / 400
        if score > best["score"]:
            best = {"angle": angle, "text": full_text, "score": score, "price": price, "raw": result}
    return best


def extract_price(text: str) -> str:
    candidates = [
        re.sub(r"\s+", "", m.group(0)).replace(",", ".").replace("'", ".").replace("`", ".").replace("’", ".")
        for m in PRICE_RE.finditer(text)
    ]
    decimal = [c for c in candidates if "." in c]
    if decimal:
        return decimal[-1]
    numbers = [int(c) for c in candidates if c.isdigit()]
    plausible = [n for n in numbers if 20 <= n <= 99999 and n not in {2025, 2026}]
    return str(plausible[-1]) if plausible else ""


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    ocr = easyocr.Reader(["ru", "en"], gpu=False)

    rows = []
    crop_dir = Path("runs/ocr/crops")
    crop_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_images(Path(args.source)):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        results = model.predict(source=str(image_path), conf=args.conf, verbose=False)
        boxes = results[0].boxes if results else []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            pad = 12
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(image.shape[1], x2 + pad)
            y2 = min(image.shape[0], y2 + pad)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_path = crop_dir / f"{image_path.stem}_{i}.jpg"
            cv2.imwrite(str(crop_path), crop)

            ocr_best = read_best_rotation(ocr, crop)
            full_text = ocr_best["text"]
            rows.append(
                {
                    "image": image_path.name,
                    "crop": str(crop_path),
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                    "det_conf": float(box.conf[0]),
                    "ocr_angle": ocr_best["angle"],
                    "ocr_text": full_text,
                    "product_name_guess": full_text,
                    "price_guess": ocr_best["price"],
                    "barcode_guess": " ".join(BARCODE_RE.findall(full_text)),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"saved {args.out} with {len(df)} rows")


if __name__ == "__main__":
    main()
