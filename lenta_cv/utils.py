from __future__ import annotations

import math
import re
from collections import Counter

import cv2
import numpy as np

from .schemas import Detection


PRICE_RE = re.compile(r"\b\d{1,5}\s*[.,]\s*\d{2}\b|\b\d{2,5}\b")
BARCODE_RE = re.compile(r"\b\d{12,14}\b")
DATE_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{4}(?:\s+\d{1,2}:\d{2})?\b")
PERCENT_RE = re.compile(r"-\d{1,3}%")
SKU_RE = re.compile(r"\b\d{9,15}\b")


def iou(box_a: Detection, box_b: Detection) -> float:
    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = box_a.area + box_b.area - intersection
    return intersection / union if union > 0 else 0.0


def crop_with_padding(image: np.ndarray, detection: Detection, pad: int = 12) -> tuple[np.ndarray, Detection]:
    height, width = image.shape[:2]
    x1 = max(0, detection.x1 - pad)
    y1 = max(0, detection.y1 - pad)
    x2 = min(width, detection.x2 + pad)
    y2 = min(height, detection.y2 + pad)
    return image[y1:y2, x1:x2].copy(), Detection(x1=x1, y1=y1, x2=x2, y2=y2, conf=detection.conf, class_id=detection.class_id)


def compute_sharpness(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness_penalty(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 1.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bright_ratio = float((gray > 245).mean())
    dark_ratio = float((gray < 20).mean())
    return max(0.0, 1.0 - bright_ratio - 0.5 * dark_ratio)


def score_crop(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    height, width = crop.shape[:2]
    area = height * width
    sharpness = compute_sharpness(crop)
    brightness_penalty = compute_brightness_penalty(crop)
    area_bonus = min(1.0, area / 200_000)
    sharpness_bonus = min(1.0, math.log1p(sharpness) / 8.0)
    aspect = width / max(1, height)
    aspect_bonus = 1.0 if 0.8 <= aspect <= 3.5 else 0.8
    return 0.45 * sharpness_bonus + 0.30 * area_bonus + 0.15 * brightness_penalty + 0.10 * aspect_bonus


def normalize_price(value: str) -> float | None:
    cleaned = value.replace(" ", "").replace(",", ".")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if 0 < number < 1_000_000:
        return number
    return None


def extract_prices(text: str) -> list[float]:
    prices: list[float] = []
    for match in PRICE_RE.finditer(text):
        value = normalize_price(match.group(0))
        if value is not None and value not in prices:
            prices.append(value)
    return prices


def extract_barcode(text: str) -> str | None:
    matches = BARCODE_RE.findall(text)
    return matches[0] if matches else None


def extract_date(text: str) -> str | None:
    match = DATE_RE.search(text)
    return match.group(0) if match else None


def extract_discount(text: str) -> str | None:
    match = PERCENT_RE.search(text)
    return match.group(0) if match else None


def extract_sku(text: str) -> str | None:
    candidates = SKU_RE.findall(text)
    if not candidates:
        return None
    return max(candidates, key=len)


def most_common_text(values: list[str]) -> str | None:
    cleaned = [value.strip() for value in values if value and value.strip()]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]

