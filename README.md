# Lenta Tech Life Hack MVP

Репозиторий содержит стартовый каркас для multi-stage решения:

`video -> price_tag detection -> track aggregation -> best frame -> OCR extraction -> CSV`

Это не финальная production-версия, но уже удобная база, на которую можно наслаивать:

- zone detector внутри ценника;
- barcode / QR decoders;
- temporal voting по нескольким кадрам;
- UI для загрузки видео и скачивания CSV.

## Что уже есть

- `scripts/build_yolo_dataset.py` - сборка датасета детекции ценников из GT CSV.
- `scripts/train_yolo.py` - обучение YOLO-детектора ценников.
- `scripts/ocr_price_tags.py` - старый черновой OCR по кропам.
- `scripts/run_video_pipeline.py` - новый end-to-end запуск по видео.
- `lenta_cv/` - модули пайплайна: схемы, трекинг, extraction, сборка CSV.

## Быстрый запуск

1. Обучить или подготовить веса детектора ценников.
2. Запустить inference:

```bash
python scripts/run_video_pipeline.py \
  --video data/videos/25_12-20.mp4 \
  --weights runs/detect/runs/price_tag_mvp/weights/best.pt \
  --out submission_inference.csv
```

Минимальные зависимости:

```bash
pip install -r requirements.txt
```

## Ключевое ограничение текущего MVP

Сейчас extraction внутри ценника всё ещё OCR-first. Архитектура уже подготовлена под следующий шаг:

`tag crop -> semantic zones -> specialized recognizers -> fusion`

