# Benchmark: эталон для клипа `data/hf_*.mp4`

Источник: [nofriendsatall/video_analytics](https://github.com/nofriendsatall/video_analytics).

| Файл | Что это |
|---|---|
| `Timings.txt` | Исходная разметка: сотрудники, события с таймингами, интервалы потребления |
| `First.png`, `Second.png` | Кадры 0 и 86 с номерами людей над головами |
| `ground_truth.json` | Машиночитаемая версия: якоря голов (координаты 1920×1080, кадры 0 и 86), роли, события, состояния |

Сравнение с эталоном: `python metrics.py --tracks data/output/tracks.json --events data/output/events.json`
(или `run_hybrid(..., gt=...)` в `track.py`). Результаты экспериментов — в [`../reports/EXPERIMENTS.md`](../reports/EXPERIMENTS.md).
