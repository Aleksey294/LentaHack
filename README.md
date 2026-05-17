# Lenta Tech Life Hack MVP

Первичная версия: baseline-сабмит, YOLO-детектор ценников и черновой OCR для названий/цен.

## Быстрый сабмит

```powershell
C:\conda\envs\osnov\python.exe scripts\download_drive_csvs.py
C:\conda\envs\osnov\python.exe make_submission.py
```

Выход: `submission.csv`.

## YOLO + OCR

```powershell
C:\conda\envs\osnov\python.exe scripts\download_drive_csvs.py --videos
C:\conda\envs\osnov\python.exe scripts\build_yolo_dataset.py
C:\conda\envs\osnov\python.exe scripts\train_yolo.py
C:\conda\envs\osnov\python.exe scripts\ocr_price_tags.py --source data\yolo_price_tags\images\val --out ocr_predictions_conf50.csv --conf 0.5
```

Выходы:

- `data/yolo_price_tags` - YOLO dataset.
- `runs/detect/runs/price_tag_mvp/weights/best.pt` - веса YOLO.
- `ocr_predictions_conf50.csv` - найденные ценники, OCR-текст, примерная цена.
- `runs/ocr/crops` - кропы ценников для визуальной проверки.

Важно: OCR пока черновой. Он уже читает часть названий и цен, но на повернутых/размытых ценниках часто ошибается.
