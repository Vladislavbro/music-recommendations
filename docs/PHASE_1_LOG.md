# Phase 1 — журнал работы

> Этот документ — **главный источник истины по ходу Phase 1**. Каждая сессия начинается с его прочтения и заканчивается обновлением. Полный план: `~/.claude/plans/zany-questing-wadler.md`. Высокоуровневая дорожная карта проекта: `CLAUDE.md`.

## Как работать с этим логом в новых чатах

В новом чате говоришь: «Продолжаем Phase 1, читай `CLAUDE.md` и `docs/PHASE_1_LOG.md`, выполни следующий незавершённый шаг». Я подхватываю состояние без re-read истории чата.

В конце сессии я **обновляю** этот файл: помечаю шаги ✅, добавляю принятые решения с обоснованием, фиксирую открытые вопросы и числа из data discovery.

## Цель Phase 1

Каркас репозитория + рабочий per-user скорер gSASRec на YAMBDA-50m + кэш персональных топ-K скоров. Без агрегаторов, без аудио, без групп. После завершения у нас на диске лежит чекпоинт модели и parquet со скорами для дальнейшего использования в Phase 2.

## Зафиксированные решения

| Решение | Обоснование |
|---|---|
| Только flavor `50m` | Согласовано с пользователем; полный 5b на Colab не уместится |
| `t = 0.75` в gBCE | Рекомендация Petrov & Macdonald 2023 для больших каталогов |
| Pandas-only data infra, без Polars | Свой контроль, не тянем зависимости от yambda; согласовано |
| Не импортируем код из `references/`, переписываем нужные куски в `src/` | Защита от изменений в апстриме; явная адаптация под наш интерфейс |
| `git mv yambda → references/yambda`, `git mv gSASRec-pytorch → references/gSASRec-pytorch` | Согласовано; реализуется в Step A |
| Фильтр событий для скорера: `event_type == 'listen' AND played_ratio_pct >= 50` | Стандарт yambda Constants.TRACK_LISTEN_THRESHOLD; implicit-сигнал, много данных |
| Аудио (14GB embeddings.parquet) **в Phase 1 не трогаем** | Скорер работает только на ID-эмбеддингах; subset делается в начале Phase 2 на Colab |
| Левое паддинг в `GSASRec`, прямая нумерация позиций (newest на L-1) | Проще `score()` — берём `hidden[:, -1, :]` без gather |
| Распределение размеров групп `{2:0.3, 3:0.4, 4:0.2, 5:0.1}` (для Phase 2) | Стандарт из AGREE/GroupIM |
| Журнал фазы (этот файл) обновляется в конце каждой сессии | Чтобы новые чаты подхватывали контекст без re-read истории |
| **`max_seq_len = 200`** (зафиксировано после Step 0) | Согласовано с пользователем; матчится с SASRec/gSASRec бенчмарками. Покрывает ~11% медианной истории YAMBDA, но recent-200 достаточно signal'а для скорера |
| **`hidden_dim = 256`, `n_heads = 4`, `n_layers = 3`** | По плану; стандарт gSASRec/yambda. Для каталога 631k items 128 теряет качество |
| **Min-popularity filter на items: `≥5 listens`** (после Step 0) | Согласовано; срежет хвост редких треков, обычно теряем 30-50% items без потери NDCG. Уменьшит embedding table |
| **`VAL_SIZE = 86400` (1 день) оставляем** | Step 0 показал 4,628 users в val — порог >1000 пройден, расширять не нужно |
| **`d_a = 128`** (audio embed dim, YAMBDA) | Получено через `HfFileSystem` range-read footer'а `embeddings.parquet`, без скачивания 14 ГБ |

## Прогресс по шагам

| Шаг | Описание | Статус | Артефакты |
|---|---|---|---|
| 0 | Data Discovery (`00_data_discovery.ipynb`) | ✅ | `notebooks/00_data_discovery.ipynb` прогнан, числа в разделе «Data discovery findings» |
| A | mv references + scaffold + requirements + .gitignore | ✅ | `src/{data,scorer,utils}/`, `artifacts/`, `references/{yambda,gSASRec-pytorch}/`, `requirements.txt`, обновлён `.gitignore` |
| B | `src/utils/{seed,caching}.py` | ✅ | `src/utils/seed.py`, `src/utils/caching.py` |
| C | `src/data/yambda_loader.py` (без audio) | ✅ | `src/data/yambda_loader.py`; cardinality на real-data сходится со Step 0 |
| D | `src/data/splits.py` (GTS на pandas) | ✅ | `src/data/splits.py`; инварианты держатся, числа сходятся со Step 0 |
| E | `notebooks/01_explore_yambda.ipynb` — интеграционный чек `src/data/*` + графики для текста ВКР (history length, item popularity Zipf, train/val/test по времени). EDA-числа НЕ дублируем — они уже в Step 0 | ✅ | `notebooks/01_explore_yambda.ipynb` прогнан; 4 PNG в `docs/figures/` |
| F | `src/scorer/gsasrec.py` | ⬜ | — |
| G | `src/scorer/gbce_loss.py` | ⬜ | — |
| H | `src/scorer/train.py` + `notebooks/02_train_gsasrec.ipynb` | ⬜ | `artifacts/gsasrec/` |
| I | `src/scorer/inference.py` + `notebooks/03_cache_user_scores.ipynb` | ⬜ | `artifacts/user_scores_cache/scores.parquet` |
| J | `src/data/group_synthesis.py` (заготовка для Phase 2) | ⬜ | — |

✅ — сделано, чекпоинт пройден  
🟨 — в процессе  
⬜ — не начато

## Data discovery findings

> Заполнено по итогам Step 0 (прогон `notebooks/00_data_discovery.ipynb` локально на M4 Pro, 2026-05-10).

- **Total events (50m, post-load):** 47,790,449
- **Event_type breakdown:**
  - listen: 46,467,212 (97.23%)
  - like: 881,456 (1.84%)
  - unlike: 312,972 (0.65%)
  - dislike: 107,776 (0.23%)
  - undislike: 21,033 (0.04%)
  - *Note:* `multi_event` отсутствует в `flat-multievent-50m` (есть только в `multi-event` flavor'ах)
- **Distribution of `played_ratio_pct`** (listen events): median = 100.0, 90p = 100.0 — почти все listen полные
- **После фильтра `listen & played_ratio_pct >= 50`:**
  - n_events: **29,439,278** (61.6% от всех listens проходят фильтр)
  - n_users: **9,209**
  - n_items (unique tracks): **631,003**
- **Per-user history length** (post-filter): median = **1798**, 95p = **11,198**, 99p = **17,296**, max = 26,959 → `max_seq_len = 200` (recent-200, покрытие ~11% медианы)
  - Уточнённая медиана **post-`min_pop≥5`** (Step E): **1758** — фильтр популярности срезает ~2.1% events и 14 user'ов с историей только на редких треках, медиана сдвигается на ~40 listens. На выбор `max_seq_len` не влияет.
- **Timestamp range:** [0, 25,999,995] → совпадает с `Constants.TEST_TIMESTAMP = 26000000 - 86400 = 25,913,600`, последние ~5 дней попадают в тест
- **GTS sanity** (val_size=86400, gap=1800):
  - users в train: 9,207
  - users в val: **4,628** ✅ (>1000, расширять `VAL_SIZE` не надо)
  - users в test: 4,600
- **Audio embed dim (`d_a`):** **128** (получено через `HfFileSystem` range-read schema footer'а `embeddings.parquet`, без скачивания 14 ГБ)
- **Estimated audio subset size:** 631,003 × 128 × float32 = ~**323 MB** (помещается в RAM целиком; в Phase 2 не нужен streaming)

### Ключевые наблюдения

1. **YAMBDA flavor naming = events count, not users.** `50m` ≈ 50M событий ≈ 9-10k users. Для перехода к большему числу users нужен `500m` (~100k) или `5b` (~1M). На Phase 1 9k users — потолок, для синтеза групп по 2-5 хватит десятков тысяч уникальных групп.
2. **Очень длинные истории.** Медиана 1798 listens/user — это музыкальный стриминг с короткими треками, не MovieLens. `max_seq_len = 200` отсекает агрессивно, но это совпадает со стандартом SASRec/gSASRec — в Phase 2 при необходимости можно сравнить с 500.
3. **n_items = 631k без фильтра по популярности.** Будем применять min-popularity filter `≥5 listens` в Step C, что должно уменьшить таблицу embedding'ов ощутимо (точное число — после применения фильтра в `yambda_loader.py`).

## Открытые вопросы (для Phase 2)

- **Ground truth для групп** в eval: union по listens / intersection / только likes? Решить после анализа размеров пересечений на discovery-стадии.
- **Стратегия audio subset**: `pyarrow.dataset` через `HfFileSystem` (range-read) vs полная загрузка в Colab vs предварительный subset на личном HF-репо. Решить в начале Phase 2.
- **Тип групп** (random / homogeneous / heterogeneous): первая итерация — только random.

## Риски и mitigation

См. раздел "Risks" в `~/.claude/plans/zany-questing-wadler.md`. Главные:
- groupby по 30M строк → фолбэк `sort_values + np.diff`
- val может оказаться пустым на 50m + 1-day val → расширить до 2 дней
- gBCE float64 — **не оптимизировать**

## История сессий

> Каждая сессия добавляет запись в конце. Формат: дата, что сделано, ключевые решения, что осталось.

### 2026-05-10 — Планирование
- Сформирован план Phase 1 в `~/.claude/plans/zany-questing-wadler.md`.
- Решено добавить **Step 0 (Data Discovery)** до написания кода — чтобы реальные числа из YAMBDA-50m информировали выбор `max_seq_len`, GTS `val_size`, candidate pool.
- Решено отложить аудио на Phase 2 (файл 14GB, в Phase 1 не нужен).
- Создан этот журнал. Выполнение плана **не начато** — стартует с нового чата.

### 2026-05-10 — Step 0 (notebook scaffolded)
- Создана директория `notebooks/`.
- Создан `notebooks/00_data_discovery.ipynb` — все ячейки готовы, прогон ручной (по согласованию с пользователем).
- Структура ноутбука: load 50m → event_type breakdown → played_ratio_pct distribution → filter listens (≥50%) → per-user history length (median/95p/99p) → timestamp + GTS sanity → audio dim через `HfFileSystem` range-read (без скачивания 14GB) → estimated audio subset → summary cell для копирования в этот лог.
- Решение по audio dim (Step 0): пробуем range-read через `HfFileSystem` + `pyarrow.ParquetFile`. Если падает (timeouts / unsupported) — `d_a = TBD` до Phase 2, как и зафиксировано в плане.

### 2026-05-10 — Step A ✅ done (scaffold)
- `yambda/` и `gSASRec-pytorch/` оказались nested git-клонами (со своими `.git`), не submodules — переехали обычным `mv` в `references/`. Поэтому шаг переименован «git mv → mv» в таблице прогресса.
- Создан scaffold: `src/{data,scorer,utils}/__init__.py`, пустой `artifacts/`. `notebooks/` и `docs/` уже существовали с предыдущих шагов.
- `requirements.txt` создан по списку из плана.
- `.gitignore` обновлён: добавлены `artifacts/`, `references/`, `.ipynb_checkpoints/`, `.hf_cache/`, `*.pt`, `*.parquet`. Убрано `docs/` (PHASE_1_LOG.md теперь поедет в репо вместе с кодом — нужно для Colab). `CLAUDE.md` остаётся в `.gitignore` по решению пользователя (пока не публикуем).
- Чекпоинт пройден: `python3 -c "import src; import src.data; import src.scorer; import src.utils"` отрабатывает.
- **TODO следующей сессии:** Step B — `src/utils/seed.py` (`set_seed(int)`, фиксирует random/np/torch/cudnn) + `src/utils/caching.py` (round-trip pickle для dict, save/load_parquet для DataFrame).

### 2026-05-10 — Steps B-D (код готов, ждёт HF-прогона)
- **Step B ✅** — `src/utils/seed.py` (`set_seed(seed, deterministic_torch=True)`: PYTHONHASHSEED + random + numpy + torch/cudnn, torch import опционален), `src/utils/caching.py` (`save_pickle/load_pickle`, `save_parquet/load_parquet` с auto-mkdir родительской директории). Sanity round-trip пройден локально.
- **Step C 🟨** — `src/data/yambda_loader.py`: `load_yambda("50m", cache_dir=...)`, `filter_listens(df, threshold_pct=50)`, `filter_min_popularity(df, min_count=5)`, `build_item_id_to_idx`, `apply_item_remap`, `subsample_users`. Логика проверена на синтетическом DataFrame; полная валидация cardinality (29.4M строк после фильтра, 9209 users) ждёт прогона на M4 Pro с HF-кэшем.
- **Step D 🟨** — `src/data/splits.py`: pandas-порт `flat_split_train_val_test` с `SplitConfig(test_timestamp=25_913_600, val_size=86_400, gap_size=1_800, drop_non_train_items=False)`. Инварианты `set(val.uid) ⊆ set(train.uid)`, дизъюнктность сегментов, путь `val_size=0` — все проходят на синтетике. Полный чекпоинт (4628 users в val) — на реальных данных.
- **Step E переформулирован** в таблице прогресса: ноутбук становится интеграционным тестом `src/data/*` + источник графиков для текста ВКР (history length, item popularity Zipf, train/val/test по времени). EDA-числа НЕ дублируем со Step 0.
- **Решение:** убрал sequential-порт `timesplit.py` в пользу flat-семантики — наша таблица flat (one-row-per-event), не лист-on-list. Семантика идентична: `train < t1`, `val ∈ [t1, t2)`, `test ≥ t2`, val/test ограничены train-users.
- **Real-data validation (выполнена в этой же сессии):** прогнал на M4 Pro локально (HF-кэш горячий, total 10.5s).
  - `filter_listens`: 29,439,278 events / 9,209 users / 631,003 items — **точно совпадает** со Step 0.
  - `filter_min_popularity(≥5)`: items 631,003 → 276,305 (-56%), events -2.1%, теряется 14 users у которых вся история на редких треках.
  - `global_temporal_split`: train 9,194 / val 4,596 / test 4,576 users (расхождение со Step 0 на ±15-30 users — потому что Step 0 считал без min-pop, ожидаемо). Инварианты val.uid⊆train.uid и test.uid⊆train.uid OK.
  - **Решение зафиксировать:** `filter_min_popularity` применяем ДО сплита (а не после). Иначе train может содержать редкие треки, отсутствующие в val/test после remap. Порядок: load → filter_listens → filter_min_popularity → build_item_id_to_idx → global_temporal_split.
- **TODO следующей сессии:**
  1. Step E: `notebooks/01_explore_yambda.ipynb` — sanity-чек loaders + 3-4 графика для ВКР.
  2. Step F: `src/scorer/gsasrec.py` — архитектура с левым паддингом.

### 2026-05-10 — Step E ✅ done (notebook прогнан, графики проверены)
- Создан `notebooks/01_explore_yambda.ipynb` (22 ячейки). Структура:
  1. Setup: `sys.path.insert(0, PROJECT_ROOT)`, создание `docs/figures/`.
  2. Pipeline через `src.data.*` в каноническом порядке (load → filter_listens → filter_min_popularity → build_item_id_to_idx → apply_item_remap → global_temporal_split). После каждого шага — сравнение с числами Step 0 через `assert` (29,439,278 events / 9,209 users / 631,003 items).
  3. GTS split + проверка инвариантов (`val.uid ⊆ train.uid`, дизъюнктность по времени).
  4. Три графика для ВКР, сохраняются в `docs/figures/` (200dpi PNG):
     - `history_length_hist.png` — log-y, с линиями `max_seq_len=200` и `median=1798`.
     - `item_popularity_zipf.png` — log-log, before/after `min_pop≥5`.
     - `gts_timeline.png` — events per day, с границами GTS.
     - `gts_timeline_zoom.png` (опционально) — zoom на последние 10 дней.
- Imports проверены локально (`python3 -c "from src.data.* import ..."` — OK). Ноутбук прогнан пользователем, все 4 графика сохранены в `docs/figures/`.
- Решение: ноутбук НЕ дублирует Step 0 — числа берутся из Step 0, ноутбук только ставит `assert` поверх loader'а.
- **Проверка графиков (визуальный анализ):**
  - `history_length_hist.png` — корректно. **Находка:** post-min_pop median = **1758** (vs 1798 post-filter-only). Различие ~40 listens объясняется потерей 2.1% events после `min_pop≥5`. Зафиксировано в разделе «Data discovery findings» как уточнение.
  - `item_popularity_zipf.png` — синяя/оранжевая кривые совпадают на rank<200k, расходятся ровно на `min_pop=5` пунктире. Визуально подтверждает: фильтр срезает только хвост (276k items vs 631k), голову не трогает.
  - `gts_timeline.png` — растущий ~300-дневный train с недельной модуляцией (выходные/будни). Val/test невидимы из-за aspect ratio (1 день vs 300). **Решение:** в текст ВКР пускаем `gts_timeline_zoom.png` как основной, `gts_timeline.png` — в приложение или убираем.
  - `gts_timeline_zoom.png` — лучший график: чёткий суточный паттерн, val/test совпадают по форме с train (нет distribution shift), 30-min gaps не разрешаются на 1-час корзинах (нормально).
- **Артефакты Step E:** 4 PNG в `docs/figures/` (всего ~280 KB), `notebooks/01_explore_yambda.ipynb` (22 ячейки).
- **TODO следующей сессии:**
  1. Step F: `src/scorer/gsasrec.py` — SASRec-архитектура с левым паддингом, `forward([B,L]) → [B,L,H]`, `score(seq, candidates) → [B,K]`. Адаптация из `references/yambda/benchmarks/models/sasrec/model.py` под наш интерфейс.

### 2026-05-10 — Step 0 ✅ done (прогон + решения)
- Пользователь прогнал `00_data_discovery.ipynb` локально, прислал summary. Числа перенесены в раздел «Data discovery findings» выше.
- **Решения, принятые на основе чисел:**
  - `max_seq_len = 200` — несмотря на медиану истории 1798, остаёмся на стандарте SASRec/gSASRec. Recent-200 достаточно signal'а; в Phase 2 при необходимости сравним с 500.
  - `hidden_dim = 256, n_heads = 4, n_layers = 3` — по плану, не меняем. На каталоге 631k items 128 теряет качество.
  - **Min-popularity filter на items: `≥5 listens`** — добавляется в `src/data/yambda_loader.py` в Step C. Уменьшит embedding table.
  - `VAL_SIZE = 86400` (1 день) оставляем — 4,628 users в val, расширять не нужно.
  - Audio range-read через `HfFileSystem` сработал → `d_a = 128`. Subset для 631k items ≈ 323 MB (помещается в RAM целиком). В Phase 2 audio loader использует `pyarrow.dataset` или per-row-group чтение с `isin`-фильтром по `item_id`.
- **Важное наблюдение:** YAMBDA flavor `50m` означает 50M событий ≈ 9-10k users (а не 50M users). Это потолок Phase 1. Если в Phase 2 нужно больше users — будет переход на `500m` (~100k users).
- **TODO следующей сессии:** Step A — `git mv yambda → references/yambda`, `git mv gSASRec-pytorch → references/gSASRec-pytorch`, scaffold `src/{data,scorer,utils}/`, `requirements.txt`, обновление `.gitignore`. Артефакт чекпоинта: `python -c "import src"` работает.
