# Phase 1 — журнал работы

> Главный источник истины по Phase 1. Новые чаты начинают с чтения `CLAUDE.md` + этого файла.

## Цель Phase 1

Каркас репозитория + рабочий per-user скорер на YAMBDA-50m + кэш персональных топ-K скоров. Без агрегаторов, без аудио, без групп. На выходе — чекпоинт модели на диске и parquet со скорами для Phase 2.

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
| Фильтр событий: `listen & played_ratio_pct ≥ 50` | Стандарт yambda `Constants.TRACK_LISTEN_THRESHOLD`; implicit-сигнал, много данных |
| `min_pop ≥ 5` применяется ДО сплита | Иначе train может содержать редкие треки, отсутствующие в val/test после remap |
| Канонический pipeline | `load → filter_listens → filter_min_popularity → build_item_id_to_idx → apply_item_remap → global_temporal_split` |
| Float64 в gBCE-трансформации позитивного логита | НЕ оптимизировать в float32 — численная стабильность при больших каталогах (n_items ~6e5) |
| Per-position loss (не per-user) | Стандарт SASRec, лучше использует длинные последовательности |
| `item_id_to_idx` рядом с чекпоинтом | Без него inference в новом ноутбуке не сможет восстановить mapping; Phase 2 тоже его читает |

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
| K | `src/data/group_synthesis.py` (заготовка под Phase 2) | ⬜ (закрыт в Phase 2 как Шаг 3) |

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

- **Total events (50m):** 47,790,449
- **Event_type breakdown:** listen 97.23%, like 1.84%, unlike 0.65%, dislike 0.23%, undislike 0.04%. `multi_event` отсутствует в `flat-multievent-50m` (есть только в `multi-event` flavor'ах).
- **Distribution of `played_ratio_pct`** (listen events): median = 100.0, 90p = 100.0 — почти все listen полные.
- **После `listen & played_ratio_pct ≥ 50`:** 29,439,278 events / **9,209 users** / **631,003 items**.
- **После `min_pop ≥ 5`:** items 631,003 → 276,305 (-56%), events -2.1%, теряется 14 users у которых вся история на редких треках.
- **Per-user history** (post-filter): median **1798** (post-min_pop: 1758), 95p 11,198, 99p 17,296, max 26,959.
- **Timestamp range:** [0, 25,999,995]. `TEST_TIMESTAMP = 26000000 - 86400 = 25,913,600`, последние ~5 дней попадают в тест.
- **GTS sanity** (val_size=86400, gap=1800): train 9,194 / val 4,596 / test 4,576 users.
- **Audio:** `d_a = 128` (через `HfFileSystem` range-read schema footer'а `embeddings.parquet`, без скачивания 14 ГБ), subset 631k items ≈ 323 MB.

### Ключевые наблюдения

1. **YAMBDA flavor naming = events count, not users.** `50m` ≈ 50M событий ≈ 9-10k users. Для перехода к большему числу users нужен `500m` (~100k) или `5b` (~1M). На Phase 1 9k users — потолок, для синтеза групп по 2-5 хватит десятков тысяч уникальных групп.
2. **Очень длинные истории.** Медиана 1798 listens/user — это музыкальный стриминг с короткими треками, не MovieLens. `max_seq_len = 200` отсекает агрессивно, но это совпадает со стандартом SASRec/gSASRec.
3. **n_items = 631k без фильтра по популярности.** Min-pop≥5 уменьшает таблицу embedding'ов до 276k (-56%), голову распределения не трогает.

## Открытые вопросы (для Phase 2)

- Ground truth для групп: union по listens / intersection / только likes?
- Audio subset: range-read vs полная загрузка в Colab vs preprocessed HF-repo.
- Тип групп: первая итерация — random.

## История сессий

### 2026-05-10 — Step 0 (Data Discovery) ✅

- Создан `notebooks/00_data_discovery.ipynb` — все ячейки готовы, прогон ручной.
- Структура: load 50m → event_type breakdown → played_ratio_pct distribution → filter listens (≥50%) → per-user history length → timestamp + GTS sanity → audio dim через `HfFileSystem` range-read.
- Audio dim: пробовали range-read через `HfFileSystem` + `pyarrow.ParquetFile`. Сработало → `d_a = 128`.
- Пользователь прогнал ноутбук локально, числа выше.
- **Решения, принятые на основе чисел:**
  - `max_seq_len = 200` — стандарт SASRec/gSASRec; recent-200 достаточно signal'а.
  - `hidden_dim = 256, n_heads = 4, n_layers = 3` — по плану (потом откатили до 64/2/2 после воспроизведения Yandex baseline).
  - **Min-popularity filter `≥5 listens`** — добавлен в `yambda_loader.py`.
  - `VAL_SIZE = 86400` (1 день) оставляем — 4,628 users в val, порог >1000 пройден.
- **Важное наблюдение:** YAMBDA flavor `50m` = 50M событий ≈ 9-10k users (не users!). Для большего числа users → `500m`.

### 2026-05-10 — Step A (scaffold) ✅

- `yambda/` и `gSASRec-pytorch/` оказались nested git-клонами (со своими `.git`), не submodules — переехали обычным `mv` в `references/`. Шаг переименован «git mv → mv» в таблице прогресса.
- Создан scaffold: `src/{data,scorer,utils}/__init__.py`, пустой `artifacts/`. `notebooks/` и `docs/` уже существовали.
- `requirements.txt` создан по списку из плана.
- `.gitignore` обновлён: добавлены `artifacts/`, `references/`, `.ipynb_checkpoints/`, `.hf_cache/`, `*.pt`, `*.parquet`. Убрано `docs/` (PHASE_1_LOG.md теперь идёт в репо вместе с кодом — нужно для Colab). `CLAUDE.md` остаётся в `.gitignore` (пока не публикуем).
- Чекпоинт: `python3 -c "import src; import src.data; import src.scorer; import src.utils"` отрабатывает.

### 2026-05-10 — Steps B-D ✅

- **Step B** — `src/utils/seed.py` (`set_seed(seed, deterministic_torch=True)`: PYTHONHASHSEED + random + numpy + torch/cudnn, torch import опционален), `src/utils/caching.py` (`save_pickle/load_pickle`, `save_parquet/load_parquet` с auto-mkdir родительской директории). Sanity round-trip пройден локально.
- **Step C** — `src/data/yambda_loader.py`: `load_yambda("50m", cache_dir=...)`, `filter_listens(df, threshold_pct=50)`, `filter_min_popularity(df, min_count=5)`, `build_item_id_to_idx`, `apply_item_remap`, `subsample_users`.
- **Step D** — `src/data/splits.py`: pandas-порт `flat_split_train_val_test` с `SplitConfig(test_timestamp=25_913_600, val_size=86_400, gap_size=1_800, drop_non_train_items=False)`. Инварианты `set(val.uid) ⊆ set(train.uid)`, дизъюнктность сегментов — все проходят.
- **Решение:** убрали sequential-порт `timesplit.py` в пользу flat-семантики — наша таблица flat (one-row-per-event), не лист-on-list. Семантика идентична: `train < t1`, `val ∈ [t1, t2)`, `test ≥ t2`, val/test ограничены train-users.
- **Real-data validation (на M4 Pro локально, HF-кэш горячий, 10.5s):**
  - `filter_listens`: 29,439,278 events / 9,209 users / 631,003 items — **точно совпадает** со Step 0.
  - `filter_min_popularity(≥5)`: items 631,003 → 276,305 (-56%), events -2.1%, теряется 14 users.
  - `global_temporal_split`: train 9,194 / val 4,596 / test 4,576 users (расхождение со Step 0 на ±15-30 users — Step 0 считал без min-pop, ожидаемо).
- **Решение зафиксировано:** `filter_min_popularity` применяется ДО сплита. Иначе train может содержать редкие треки, отсутствующие в val/test после remap. Канонический порядок: `load → filter_listens → filter_min_popularity → build_item_id_to_idx → global_temporal_split`.

### 2026-05-10 — Step E ✅

- Создан `notebooks/01_explore_yambda.ipynb` (22 ячейки). Структура:
  1. Setup: `sys.path.insert(0, PROJECT_ROOT)`, создание `docs/figures/`.
  2. Pipeline через `src.data.*` в каноническом порядке. После каждого шага — сравнение с числами Step 0 через `assert`.
  3. GTS split + проверка инвариантов (`val.uid ⊆ train.uid`, дизъюнктность по времени).
  4. Четыре графика для ВКР (200 dpi PNG в `docs/figures/`):
     - `history_length_hist.png` — log-y, с линиями `max_seq_len=200` и `median=1798`.
     - `item_popularity_zipf.png` — log-log, before/after `min_pop≥5`.
     - `gts_timeline.png` — events per day, с границами GTS.
     - `gts_timeline_zoom.png` — zoom на последние 10 дней.
- **Решение:** ноутбук НЕ дублирует Step 0 — числа берутся из Step 0, ноутбук только ставит `assert` поверх loader'а.
- **Проверка графиков (визуальный анализ):**
  - `history_length_hist.png` — корректно. **Находка:** post-min_pop median = **1758** (vs 1798 post-filter-only). Различие ~40 listens объясняется потерей 2.1% events после `min_pop≥5`. Зафиксировано как уточнение в data discovery findings.
  - `item_popularity_zipf.png` — синяя/оранжевая кривые совпадают на rank<200k, расходятся ровно на `min_pop=5` пунктире. Визуально подтверждает: фильтр срезает только хвост (276k items vs 631k), голову не трогает.
  - `gts_timeline.png` — растущий ~300-дневный train с недельной модуляцией (выходные/будни). Val/test невидимы из-за aspect ratio (1 день vs 300).
  - `gts_timeline_zoom.png` — лучший график: чёткий суточный паттерн, val/test совпадают по форме с train (нет distribution shift), 30-min gaps не разрешаются на 1-час корзинах.
- **Решение для текста ВКР:** в основной текст пускаем `gts_timeline_zoom.png`, `gts_timeline.png` — в приложение или убираем.

### 2026-05-10 — Steps F-G ✅

- **Step F** — `src/scorer/gsasrec.py`. Архитектура близка к yambda SASRec, но проще:
  - **Left padding, прямые позиции 0..L-1** (newest на L-1) — `score()` берёт `hidden[:, -1, :]` без gather по длинам, снимает целый класс багов с offsets/cumsum.
  - `nn.Embedding(n_items+1, H, padding_idx=0)` — padding-idx занулён инициализацией и не обучается (PyTorch это гарантирует).
  - `nn.TransformerEncoder` (`batch_first=True`) с causal mask `triu(diag=1)` и `src_key_padding_mask = (seq == 0)`. Guard на полностью-пустые строки (kpm=False), чтобы избежать NaN в attention — на реальных батчах не сработает, но защищает unit-тесты.
  - Init: trunc_normal(std=0.02) для weights, ones для norm, zeros для bias — как в yambda.
  - Loss НЕ внутри модели (в отличие от yambda), считается снаружи через `gbce_loss` — модель используется и в train, и в `inference.py`.
- **Step G** — `src/scorer/gbce_loss.py`. Прямой порт строк 59-72 из `references/gSASRec-pytorch/train_gsasrec.py`, обёрнутый в функцию по сигнатуре из CLAUDE.md.
  - Сигнатура `gbce_loss(pos_logits, neg_logits, n_items, n_neg, t=0.75)`. Поддерживает broadcast: pos `[B]` / neg `[B, n_neg]` (per-step), и pos `[B, L]` / neg `[B, L, n_neg]` (seq2seq) — `pos.unsqueeze(-1)` приводит формы к `[..., 1]` и `[..., n_neg]`.
  - Float64 для трансформации позитивного логита (clamp с eps), затем concat и `F.binary_cross_entropy_with_logits(reduction="mean")`. Возврат — в исходном dtype pos.
  - **НЕ оптимизировать float64 → float32** (зафиксировано в Risks): теряется численная стабильность при больших каталогах (n_items ~6e5).
- **Smoke-тест прошёл** на `torch 2.x`:
  - `forward [B=8, L=50] → [8, 50, 64]`, все finite. Left-pad на row 0 (10 паддингов) обработан без NaN.
  - `score(seq, cand[8,16]) → [8, 16]`, finite.
  - `gbce_loss` на flat и seq2seq формах — finite.
  - `gbce(t=0) = 0.745660` **точно совпадает** с `F.binary_cross_entropy_with_logits` на тех же логитах — подтверждает, что при `t=0` восстанавливается обычный BCE (как заявлено Petrov & Macdonald).
  - Backward через модель: loss finite, **grad на `item_embeddings.weight[0]` (padding-row) = 0.0** — `padding_idx=0` корректно изолирует пад-эмбеддинг от обучения. Все 28 обучаемых параметров получают ненулевой grad.

### 2026-05-10 — Steps H-I (изначальный конфиг, до Yandex-репро) ✅

- **Step H** — `src/scorer/train.py` (~270 строк):
  - `build_user_sequences(df, max_seq_len)`: dict[uid → np.ndarray], сортировка по timestamp asc, последние `max_seq_len+1` events. Юзеры с <2 events отбрасываются (нет seq2seq target).
  - `compute_item_popularity(df, n_items, smoothing=0.75)`: word2vec-style сглаживание `count^0.75 / sum`, padding-row=0.
  - `SeqTrainDataset` + `make_train_collate(max_seq_len)`: collate левым паддингом — `inputs = seq[:-1]`, `targets = seq[1:]`, оба `[B, L]`. Маска padding'а: `targets != PAD_IDX`.
  - `PopularityNegativeSampler`: `np.random.default_rng().choice` (быстрее `torch.multinomial` для CPU+большой каталог). Коллизии с позитивом игнорируются — при каталоге 276k и n_neg=256 P(коллизии)<0.1%.
  - Loss-flow: `hidden = model(inputs)` → `flat_h = hidden[mask]` → embed positives/negatives → `pos_logits = (flat_h * pos_emb).sum(-1)`, `neg_logits = einsum("nh,nkh->nk", flat_h, neg_emb)` → `gbce_loss(pos_logits, neg_logits, n_items, n_neg, t=0.75)`. Усреднение per-position (не per-user) — корректно для seq2seq.
  - `evaluate_ndcg(...)`: full-catalog NDCG@K на val users, изначально с маскированием train-истории (потом убрано при Yandex-репро). По умолчанию батч 64, `topk` на GPU, NDCG/IDCG в Python.
- **Step I** — `src/scorer/inference.py` (~80 строк):
  - `load_checkpoint(path)`: восстанавливает архитектуру из `cfg` внутри чекпоинта (без передачи параметров наружу).
  - `cache_user_scores(model, sequences, max_seq_len, n_items, out_path, cfg)`: батчами строит `last = h[:, -1, :]`, скорит полным каталогом (`last @ item_emb.T`), исключает PAD_IDX и (опционально) train-историю, берёт `topk K`, пишет parquet `(uid, item_idx, score, rank)`. Дефолт `K=200`.
- **Ноутбуки (изначально раздельные):**
  - `02_train_gsasrec.ipynb`: setup → канонический pipeline → `train_gsasrec(...)` с full-config → загрузка чекпоинта + сохранение `item_id_to_idx.pkl` → график train_loss/val_NDCG. Флаг `SMOKE` для подсэмпла 500 users на M4 Pro.
  - `03_cache_user_scores.ipynb`: тот же pipeline, читает сохранённый `item_id_to_idx.pkl` (гарантия совпадения индексов), прогоняет `cache_user_scores` с `K=200`.
- **Smoke-тест на CPU** (50 users / 200 items / 32 hidden / 2 epoch): training run прошёл (loss=0.677→0.664), val_ndcg@10=0.017 на random-данных (>0, sane), checkpoint сохранён, inference дал 50 users × K=10 строк в parquet.
- **Решения:**
  - Per-position loss (не per-user) — стандарт SASRec, лучше использует длинные последовательности.
  - Popularity smoothing 0.75 — word2vec-style, отдельный параметр в `TrainConfig`.
  - `item_id_to_idx` сохраняем рядом с чекпоинтом — без него inference в новом ноутбуке не сможет восстановить mapping. Phase 2 тоже его читает.
  - `_left_pad` помечен как private, но переиспользуется в `inference.py` — допустимо, оба модуля принадлежат scorer-пакету.

### 2026-05-10 — Hotfix MixedNegativeSampler

После первого Colab-прогона на G4: `epoch 0/1/2: train_loss быстро падает до 0.008, val_ndcg@10 ≈ 0.0003 ≈ random`.

**Диагноз — popularity shortcut:** 256 negatives из `pop^0.75` всегда дают одни и те же топ-чарт треки → модель учит «жанр в истории = хорошо, попса = плохо», fine-grained next-item не выучивается.

**Правка:** добавлен `MixedNegativeSampler` (popularity + uniform), дефолты `TrainConfig`: `n_neg=512, mix_uniform=0.5`. Smoke на CPU прошёл. Ноутбук 02 обновлён.

Это был промежуточный шаг — финально на репродукции Yandex baseline всё равно сошлись к `n_neg=1, mix_uniform=1.0` (чистый uniform).

### 2026-05-11 — Воспроизведение Yandex baseline ✅

- Серия итераций (см. таблицу выше): loss → размер модели → gBCE → eval-протокол.
- Финальный шаг — убрать маскирование истории из `evaluate_ndcg` (строки 263-265 в `train.py` закомментированы). NDCG@10 0.0229 → 0.0726.
- **Объединил ноутбуки 03+04 в один**: train+cache теперь в `03_cache_user_scores.ipynb`, сразу сохраняет всё нужное для Phase 2.
- **Латентный эффект на bestcheckpoint:** до фикса архитектурного бага с pad (в Phase 2 Шаг 1) `evaluate_ndcg` тоже страдал → короткие юзеры давали 0 contribution. Baseline 0.0726 слегка занижен; re-eval не делаю, для текста ВКР — footnote.
- Phase 1 закрыт. Осталось — Step K (group synthesis заготовка) при старте Phase 2.

## Риски (не реализовавшиеся)

- groupby по 30M строк → фолбэк `sort_values + np.diff` (не понадобился, pandas справился).
- val пустой → не случилось (4,596 users).
- gBCE float64 → не оптимизировали, не понадобилось.
