# Lenta Tech Life Hack MVP

Первичная версия: baseline-сабмит, YOLO-детектор ценников и черновой OCR для названий/цен.

Выход: `submission.csv`.

## YOLO + OCR
Выходы:

- `data/yolo_price_tags` - YOLO dataset.
- `runs/detect/runs/price_tag_mvp/weights/best.pt` - веса YOLO.
- `ocr_predictions_conf50.csv` - найденные ценники, OCR-текст, примерная цена.
- `runs/ocr/crops` - кропы ценников для визуальной проверки.
