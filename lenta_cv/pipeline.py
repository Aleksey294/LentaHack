from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from .extractors import NullTagExtractor, OCRTagExtractor
from .schemas import Detection, ExtractedFields, TrackSample
from .tracking import SimpleIOUTracker
from .utils import crop_with_padding, most_common_text, score_crop


SUBMISSION_COLUMNS = [
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]


class PriceTagPipeline:
    def __init__(
        self,
        detector_weights: str | Path,
        detector_conf: float = 0.20,
        frame_stride: int = 3,
        iou_threshold: float = 0.30,
        max_misses: int = 8,
    ) -> None:
        detector_weights = Path(detector_weights)
        if not detector_weights.exists():
            raise FileNotFoundError(f"Detector weights not found: {detector_weights}")
        self.detector = YOLO(str(detector_weights))
        self.detector_conf = detector_conf
        self.frame_stride = frame_stride
        self.tracker = SimpleIOUTracker(iou_threshold=iou_threshold, max_misses=max_misses)
        try:
            self.extractor = OCRTagExtractor()
        except RuntimeError:
            self.extractor = NullTagExtractor()

    def process_video(self, video_path: str | Path, crop_dir: str | Path | None = None) -> pd.DataFrame:
        video_path = Path(video_path)
        crop_dir_path = Path(crop_dir) if crop_dir else None
        if crop_dir_path is not None:
            crop_dir_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % self.frame_stride == 0:
                samples = self._detect_frame(frame, frame_index, fps, crop_dir_path, video_path.stem)
                self.tracker.update(samples)

            frame_index += 1

        cap.release()
        tracks = self.tracker.finish()
        rows = [self._track_to_row(track, video_path.name) for track in tracks if track.samples]
        df = pd.DataFrame(rows)

        for column in SUBMISSION_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA
        return df[SUBMISSION_COLUMNS]

    def _detect_frame(
        self,
        frame,
        frame_index: int,
        fps: float,
        crop_dir: Path | None,
        video_stem: str,
    ) -> list[TrackSample]:
        results = self.detector.predict(source=frame, conf=self.detector_conf, verbose=False)
        boxes = results[0].boxes if results else []
        timestamp_ms = int(frame_index * 1000.0 / fps)
        samples: list[TrackSample] = []

        for box_index, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            detection = Detection(x1=x1, y1=y1, x2=x2, y2=y2, conf=float(box.conf[0]))
            crop, padded_detection = crop_with_padding(frame, detection, pad=12)
            crop_score = score_crop(crop)
            sample = TrackSample(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                detection=padded_detection,
                crop_score=crop_score,
            )

            if crop_dir is not None and crop.size != 0:
                crop_path = crop_dir / f"{video_stem}_f{frame_index:06d}_b{box_index:02d}.jpg"
                cv2.imwrite(str(crop_path), crop)
                sample.crop_path = crop_path

            samples.append(sample)

        return samples

    def _track_to_row(self, track, filename: str) -> dict:
        best_sample = track.best_sample
        if best_sample.crop_path is not None:
            crop = cv2.imread(str(best_sample.crop_path))
        else:
            raise RuntimeError("Best sample crop path is missing. Enable crop_dir for extraction.")
        if crop is None:
            raise RuntimeError(f"Cannot read crop: {best_sample.crop_path}")

        primary_fields = self.extractor.extract_fields(crop)
        fused_fields = self._fuse_fields(track.samples, primary_fields)
        row = asdict(fused_fields)
        row.update(
            {
                "filename": filename,
                "frame_timestamp": best_sample.timestamp_ms,
                "x_min": best_sample.detection.x1,
                "y_min": best_sample.detection.y1,
                "x_max": best_sample.detection.x2,
                "y_max": best_sample.detection.y2,
            }
        )
        row.pop("raw_text", None)
        row.pop("raw_ocr_confidence", None)
        return row

    def _fuse_fields(self, samples: list[TrackSample], primary_fields: ExtractedFields) -> ExtractedFields:
        if len(samples) <= 1:
            return primary_fields

        text_candidates: list[str] = []
        confidence_candidates: list[float] = []

        for sample in samples[: min(len(samples), 5)]:
            if sample.crop_path is None:
                continue
            crop = cv2.imread(str(sample.crop_path))
            if crop is None:
                continue
            fields = self.extractor.extract_fields(crop)
            if fields.raw_text:
                text_candidates.append(fields.raw_text)
            if fields.raw_ocr_confidence is not None:
                confidence_candidates.append(fields.raw_ocr_confidence)

        best_text = most_common_text(text_candidates)
        if best_text:
            primary_fields.raw_text = best_text
        if confidence_candidates:
            primary_fields.raw_ocr_confidence = max(confidence_candidates)
        return primary_fields
