# FlowMetric

Видеоаналитика зала кафе: детекция и трекинг людей, роли «сотрудник / посетитель» и события обслуживания по видео с камеры.

Гибридный подход:
1. **Локально на каждом кадре:** YOLO11s @1280 + BoT-SORT дают боксы и покадровые ID.
2. **[GPT-6 Astra](https://vsellm.ru/models/openai/gpt-6-astra) раз в секунду:** по батчам из 16 кадров даёт переидентификацию людей и их роль.
3. **Синхронизация по IoU:** фрагменты локальных треков склеиваются в стабильные ID людей.
4. **События по траекториям** с зонами камеры: вход, выход, выход из кухни, посадка за стол, начало и конец обслуживания, ожидание.

На выходе получаются видео с разметкой, `tracks.json`, `events.json` и отчёт сравнения с эталоном.

## Структура

```text
track.py                  пайплайн: детекция+трекинг → Astra → синхронизация ID → события → рендер
astra.py                  клиент GPT-6 Astra (VseLLM / OpenAI), JSON-схема ответа
events.py                 события и состояния по траекториям
metrics.py                сравнение с эталоном (ID, роли, события P/R/F1, tIoU состояний)
notebook.ipynb            запуск гибрида и локального режима, сравнение
config/                   camera.json (зоны: дверь, кухня), botsort_reid.yaml
benchmark/                эталон: Timings.txt, ground_truth.json, кадры с якорями
experiments/              сетка экспериментов (детекторы × трекеры × ReID × VLM)
reports/                  результаты экспериментов (EXPERIMENTS.md, CSV)
weights/                  веса моделей (не в git, скачиваются автоматически)
data/                     входное видео; output/ и experiments/ — результаты прогонов
```

## Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "VSELLM_API_KEY=..." > .env      # ключ VseLLM (или OPENAI_API_KEY)
```

## Запуск

```bash
python track.py                       # гибрид: data/output/tracked.mp4 + tracks/events/benchmark_report.json
python track.py --no-astra            # только локальный трекинг
python metrics.py                     # пересчитать метрики по data/output/{tracks,events}.json
```

Ответ Astra кэшируется в `data/output/astra_reid.json`. Чтобы запросить модель заново, удалите этот файл.

## Результаты

Лучшая конфигурация — YOLO11s @1280 + BoT-SORT + Astra 1 fps:

| Метрика | Значение |
|---|---|
| F1 событий | 0.53 (recall 0.89) |
| Непрерывность ID | 0.92 |
| Число ID | 34 на 30 человек |
| Точность ролей | 100% |

Только локальный трекинг даёт F1 0.29 и непрерывность 0.68. Сравнение 138 конфигураций (YOLO, RF-DETR, 9 трекеров, ReID, варианты VLM) — в [`reports/EXPERIMENTS.md`](reports/EXPERIMENTS.md).

Чтобы повторить эксперименты, нужны дополнительные зависимости из `requirements-experiments.txt`; порядок установки описан в отчёте.
