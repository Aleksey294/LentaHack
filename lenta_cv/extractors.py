from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .schemas import ExtractedFields
from .utils import extract_barcode, extract_date, extract_discount, extract_prices, extract_sku


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported angle: {angle}")


def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]
    scale = max(2, min(5, int(900 / max(height, width)) + 1))
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.bilateralFilter(gray, 5, 35, 35)


def normalize_ocr_result(result: list) -> list[str]:
    texts: list[str] = []
    for item in result:
        if len(item) >= 2:
            text = str(item[1]).strip()
            if text:
                texts.append(text)
    return texts


def ocr_confidence(result: list) -> float:
    scores: list[float] = []
    for item in result:
        if len(item) >= 3:
            try:
                scores.append(float(item[2]))
            except (TypeError, ValueError):
                continue
    return sum(scores) / len(scores) if scores else 0.0


@dataclass(slots=True)
class OCRReadResult:
    text: str
    confidence: float
    angle: int


class OCRTagExtractor:
    def __init__(self) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("easyocr is required for OCRTagExtractor") from exc
        self.reader = easyocr.Reader(["ru", "en"], gpu=False)

    def read(self, crop: np.ndarray) -> OCRReadResult:
        best = OCRReadResult(text="", confidence=0.0, angle=0)
        for angle in (0, 90, 180, 270):
            rotated = rotate_image(crop, angle)
            prepared = preprocess_for_ocr(rotated)
            result = self.reader.readtext(prepared, detail=1, paragraph=False)
            text = " ".join(normalize_ocr_result(result))
            conf = ocr_confidence(result)
            if conf >= best.confidence:
                best = OCRReadResult(text=text, confidence=conf, angle=angle)
        return best

    def extract_fields(self, crop: np.ndarray) -> ExtractedFields:
        ocr_result = self.read(crop)
        prices = extract_prices(ocr_result.text)

        fields = ExtractedFields(
            raw_text=ocr_result.text,
            raw_ocr_confidence=ocr_result.confidence,
            barcode=extract_barcode(ocr_result.text),
            qr_code_barcode=extract_barcode(ocr_result.text),
            discount_amount=extract_discount(ocr_result.text),
            print_datetime=extract_date(ocr_result.text),
            id_sku=extract_sku(ocr_result.text),
        )

        if prices:
            fields.price_default = max(prices)
            fields.price_card = min(prices)
            if len(prices) >= 3:
                middle = sorted(prices)[1]
                fields.price_discount = middle

        cleaned_lines = [part.strip() for part in ocr_result.text.split("  ") if part.strip()]
        if cleaned_lines:
            fields.product_name = max(cleaned_lines, key=len)
        return fields


class NullTagExtractor:
    """Fallback extractor for environments without OCR dependencies."""

    def extract_fields(self, crop: np.ndarray) -> ExtractedFields:
        _ = crop
        return ExtractedFields()
