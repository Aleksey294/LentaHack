from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    class_id: int = 0

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass(slots=True)
class FrameDetection:
    frame_index: int
    timestamp_ms: int
    detection: Detection


@dataclass(slots=True)
class TrackSample:
    frame_index: int
    timestamp_ms: int
    detection: Detection
    crop_score: float
    ocr_score: float = 0.0
    crop_path: Path | None = None


@dataclass(slots=True)
class ExtractedFields:
    product_name: str | None = None
    price_default: float | None = None
    price_card: float | None = None
    price_discount: float | None = None
    barcode: str | None = None
    discount_amount: str | None = None
    id_sku: str | None = None
    print_datetime: str | None = None
    code: str | None = None
    additional_info: str | None = None
    color: str | None = None
    special_symbols: str | None = None
    qr_code_barcode: str | None = None
    price1_qr: float | None = None
    price2_qr: float | None = None
    price3_qr: float | None = None
    price4_qr: float | None = None
    wholesale_level_1_count: str | None = None
    wholesale_level_1_price: float | None = None
    wholesale_level_2_count: str | None = None
    wholesale_level_2_price: float | None = None
    action_price_qr: str | None = None
    action_code_qr: str | None = None
    raw_text: str | None = None
    raw_ocr_confidence: float | None = None


@dataclass(slots=True)
class TrackState:
    track_id: int
    samples: list[TrackSample] = field(default_factory=list)
    misses: int = 0

    def add_sample(self, sample: TrackSample) -> None:
        self.samples.append(sample)
        self.misses = 0

    @property
    def last_detection(self) -> Detection:
        return self.samples[-1].detection

    @property
    def best_sample(self) -> TrackSample:
        return max(self.samples, key=lambda item: (item.crop_score, item.ocr_score, item.detection.conf))

