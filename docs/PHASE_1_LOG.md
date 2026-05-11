# Phase 1 — журнал работы

> Главный источник истины по Phase 1. Новые чаты начинают с чтения `CLAUDE.md` + этого файла.

## Цель Phase 1

Каркас репозитория + рабочий per-user скорер на YAMBDA-50m + кэш персональных топ-K скоров. Без агрегаторов, без аудио, без групп.

## Финальная конфигурация скорера

После серии итераций (см. «Воспроизведение Yandex baseline» ниже) **отказались от gSASRec в пользу plain SASRec с BCE-loss** — конфигом, идентичным яндексовскому baseline.

| Параметр | Значение | Источник |
|---|---|---|
| Loss | BCE с 1 uniform негативом | как у Yandex; gBCE дала setup-gap, не помогла |
| `n_neg` | 1 | как у Yandex |
| `mix_uniform` | 1.0 (чистый uniform) | как у Yandex |
| `gbce_t` | 0.0 | формально ставим, эффективно отключает gBCE-калибровку |
| `hidden_dim` | 64 | яндексовский дефолт (наш 256 не давал преимущества) |
| `n_heads` | 2 | яндексовский дефолт |
| `n_layers` | 2 | яндексовский дефолт |
| `dropout` | 0.2 | оставили (у Яндекса 0.0) |
| `max_seq_len` | 200 | стандарт SASRec |
| `lr` | 1e-3 | стандарт |
| **eval** | **full-catalog NDCG@K БЕЗ маскирования истории** | как у Yandex; ключевое отличие для музыки |
| Фильтр | `listen & played_ratio_pct≥50`, `min_pop≥5` | стандарт yambda |
| Split | GTS, `val_size=86400`, `gap=1800`, `test_ts=25_913_600` | стандарт yambda |

**Итог:** test NDCG@10 = **0.0726** на Listen+ (vs Yandex 0.0742, разница в пределах шума → setup-gap закрыт).

## Зафиксированные решения

| Решение | Обоснование |
|---|---|
| Только flavor `50m` | Полный 5b на Colab не уместится |
| Pandas-only data infra | Свой контроль, не тянем Polars из yambda |
| Не импортируем из `references/`, переписываем в `src/` | Защита от изменений в апстриме |
| Аудио (14GB) в Phase 1 не трогаем | Скорер работает только на ID |
| Левый паддинг, прямые позиции 0..L-1 | Проще `score()`: `hidden[:, -1, :]` без gather |
| `d_a = 128` | Получено через `HfFileSystem` range-read, без скачивания |

## Прогресс по шагам

| Шаг | Описание | Статус |
|---|---|---|
| 0 | Data Discovery (`00_data_discovery.ipynb`) | ✅ |
| A | mv references + scaffold + requirements + .gitignore | ✅ |
| B | `src/utils/{seed,caching}.py` | ✅ |
| C | `src/data/yambda_loader.py` | ✅ |
| D | `src/data/splits.py` (GTS на pandas) | ✅ |
| E | `notebooks/01_explore_yambda.ipynb` + графики ВКР | ✅ |
| F | `src/scorer/gsasrec.py` (архитектура) | ✅ |
| G | `src/scorer/gbce_loss.py` | ✅ (оставлен в коде, но не используется в финальном конфиге) |
| H | `src/scorer/train.py` + `02_train_gsasrec.ipynb` | ✅ |
| I+J | `src/scorer/inference.py` + объединённый `03_cache_user_scores.ipynb` (train+cache в одном) | ✅ |
| K | `src/data/group_synthesis.py` (заготовка под Phase 2) | ⬜ |

## Воспроизведение Yandex baseline 

Главный результат Phase 1 — не сам скорер, а **диагностика setup-gap**. Начинали с 0.0117, доехали до 0.0726. Полезно для текста ВКР.

| Шаг | Конфиг | test NDCG@10 |
|---|---|---|
| Старт (gSASRec full) | n_neg=512, mix_uniform=0.5, gBCE t=0.75, 256/4/3, **mask history** | 0.0117 |
| Упростили loss | n_neg=1, mix_uniform=1.0 | 0.0229 |
| Урезали модель | + 64/2/2 | 0.0229 |
| Выключили gBCE | + gbce_t=0.0 | 0.0229 |
| **Убрали маскирование истории** | финальный конфиг | **0.0726** ✅ |

**Главное открытие:** в музыке нельзя маскировать историю при ранжировании. Люди переслушивают треки, и в test-таргете часто лежат item'ы из истории. Стандартный movies/books eval-протокол занижает NDCG втрое. Это сильный аргумент в текст ВКР: domain-specific особенность повторного потребления.

**Второе:** gBCE с `t=0.75` на YAMBDA не помогла. При `n_neg=1` калибровочная трансформация позитивного логита эффективно эквивалентна плейн BCE; при больших `n_neg` мы получали popularity-shortcut. Plain BCE+uniform работает не хуже и проще.

## Data discovery findings (Step 0)

- **Total events (50m):** 47,790,449. `listen`: 97.23%, `like`: 1.84%, прочее: <1%.
- **После `listen & played_ratio_pct ≥ 50`:** 29,439,278 events / **9,209 users** / **631,003 items**.
- **После `min_pop ≥ 5`:** items 631,003 → 276,305 (-56%), events -2.1%, теряется 14 users.
- **Per-user history** (post-filter): median **1798** (post-min_pop: 1758), 95p 11,198, 99p 17,296.
- **Timestamp range:** [0, 25,999,995]. `TEST_TIMESTAMP = 25,913,600`.
- **GTS sanity** (val_size=86400, gap=1800): train 9,194 / val 4,596 / test 4,576 users.
- **Audio:** `d_a = 128`, subset 631k items ≈ 323 MB.

**Важное:** YAMBDA flavor `50m` = 50M событий ≈ 9-10k users (НЕ users). Потолок Phase 1. Для большего числа users → `500m`.

## Открытые вопросы (для Phase 2)

- Ground truth для групп: union по listens / intersection / только likes?
- Audio subset: range-read vs полная загрузка в Colab vs preprocessed HF-repo.
- Тип групп: первая итерация — random.

## История сессий (краткая)

### 2026-05-10 — Steps 0-A
- Прогон `00_data_discovery.ipynb`, числа выше.
- `yambda/` и `gSASRec-pytorch/` переехали в `references/` (были nested git-клоны, не submodules → обычный mv).
- Scaffold `src/{data,scorer,utils}/`, `requirements.txt`, `.gitignore` (artifacts, references, .pt, .parquet).

### 2026-05-10 — Steps B-E
- `src/utils/{seed,caching}.py`, `src/data/yambda_loader.py`, `src/data/splits.py` (pandas-порт `flat_split_train_val_test`).
- **Решение:** `filter_min_popularity` применяется ДО сплита (иначе train может содержать редкие треки, отсутствующие в val/test после remap).
- Канонический pipeline: `load → filter_listens → filter_min_popularity → build_item_id_to_idx → apply_item_remap → global_temporal_split`.
- `notebooks/01_explore_yambda.ipynb` — sanity-чек loaders + 4 графика в `docs/figures/`.

### 2026-05-10 — Steps F-G
- `src/scorer/gsasrec.py`: SASRec с левым паддингом, прямыми позициями, `padding_idx=0`, trunc_normal init.
- `src/scorer/gbce_loss.py`: порт строк 59-72 из `references/gSASRec-pytorch/train_gsasrec.py`. Float64 для трансформации позитивного логита (не оптимизировать). Smoke подтвердил `gbce(t=0) ≡ BCE`.

### 2026-05-10 — Steps H-I (изначальный конфиг)
- `src/scorer/train.py`: seq2seq targets, popularity-based negatives с smoothing 0.75, `MixedNegativeSampler` (mix popularity+uniform), `evaluate_ndcg` с маскированием истории, early stopping.
- `src/scorer/inference.py` + ноутбуки 02/03 (изначально раздельные).
- Smoke на CPU прошёл.

### 2026-05-11 — Воспроизведение Yandex baseline
- Серия итераций (см. таблицу выше): loss → размер модели → gBCE → eval-протокол.
- Финальный шаг — убрать маскирование истории из `evaluate_ndcg` (строки 263-265 в `train.py` закомментированы). NDCG@10 0.0229 → 0.0726.
- **Объединил ноутбуки 03+04 в один**: train+cache теперь в `03_cache_user_scores.ipynb`, сразу сохраняет всё нужное для Phase 2.
- Phase 1 закрыт. Осталось — Step K (group synthesis заготовка) при старте Phase 2.

## Риски (не реализовавшиеся)

- groupby по 30M строк → фолбэк `sort_values + np.diff` (не понадобился, pandas справился).
- val пустой → не случилось (4,596 users).
- gBCE float64 → не оптимизировали, не понадобилось.
