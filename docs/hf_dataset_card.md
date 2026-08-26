---
license: apache-2.0
tags:
- recsys
- group-recommendation
- music
---

# Group aggregators over a frozen scorer — artifacts (YAMBDA-50m)

Чекпоинты и промежуточные артефакты для групповых музыкальных рекомендаций:
per-user скорер, кэш его топ-200, аудиоэмбеддинги каталога и обученные
групповые агрегаторы. Всё производное от
[`yandex/yambda`](https://huggingface.co/datasets/yandex/yambda), flavor `50m`.

## Состав

| Путь | Что это |
|---|---|
| `gsasrec/best.pt`, `config.json`, `metrics.csv` | SASRec-скорер, 276 305 items, test NDCG@10 = 0.0726 |
| `gsasrec/item_id_to_idx.pkl` | `item_id` YAMBDA → компактный индекс |
| `user_scores_cache/scores.parquet` | топ-200 скоров на пользователя, 9 170 пользователей |
| `audio/embeddings.npy` | аудиоэмбеддинги YAMBDA, 276 306 × 128, переиндексированные под каталог |
| `audio/user_profiles.npy`, `uid_to_row.pkl`, `user_audio_valid.npy` | аудиопрофили пользователей и маска покрытия |
| `aggregators/{agree,groupim,audio_agree,group_cross_attn}/` | чекпоинты групповых агрегаторов |
| `aggregators/groups_split.pkl` | синтетический split групп (10k/2k/2k, размеры 2–5, seed 42) |
| `eval_results/` | `summary.csv`, `summary_by_size.csv`, `paired.csv`, `per_sample.npz`, LaTeX-таблица |

## Результаты (групповой NDCG на test-группах)

| Метод | NDCG@10 | NDCG@20 |
|---|---:|---:|
| GroupCrossAttn | **0.0916** | **0.1024** |
| Audio-AGREE | 0.0901 | 0.1014 |
| Max Pleasure | 0.0828 | 0.0915 |
| GroupIM | 0.0811 | 0.0912 |
| AGREE | 0.0790 | 0.0901 |
| Average | 0.0695 | 0.0819 |
| Least Misery | 0.0312 | 0.0363 |

Скорер заморожен, обучаются только параметры агрегатора. Кандидаты берутся из
кэша топ-200, поэтому числа сравнимы между методами, но усечены по построению.

## Лицензия и атрибуция

Apache 2.0, унаследована от исходного датасета. `audio/embeddings.npy` —
отфильтрованная и переиндексированная копия аудиоэмбеддингов YAMBDA
(Copyright 2025 YANDEX LLC). YAMBDA опубликован для научных и исследовательских
целей; эти производные распространяются на тех же условиях.

Препринт датасета: https://arxiv.org/abs/2505.22238
