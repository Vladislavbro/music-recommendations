# Phase 2 — журнал работы

> Главный источник истины по Phase 2. Новые чаты начинают с чтения `CLAUDE.md` + `PHASE_1_LOG.md` + этого файла.

## Цель Phase 2

Обучить и сравнить 4 групповых агрегатора поверх замороженного SASRec-скорера из Phase 1: **ID-based AGREE**, **GroupIM**, **Audio-AGREE**, **Group cross-attention**. Основная метрика — NDCG@10/20 на синтетических random-группах. Конечный артефакт — таблица сравнения для текста ВКР.

Гипотеза H1 (минимальная): audio-aware агрегаторы (Audio-AGREE, GroupCrossAttn) ≥ ID-based (AGREE, GroupIM) по NDCG@10/20.

## Что готово из Phase 1 (входы)

| Артефакт | Путь | Что внутри |
|---|---|---|
| Чекпоинт SASRec | `artifacts/gsasrec/best.pt` | hidden=64, 2 layers, 2 heads, n_items=276305 |
| Конфиг + метрики | `artifacts/gsasrec/{config.json,metrics.csv}` | best val NDCG@10 ≈ 0.0855 (epoch 51), test 0.0726 |
| Карта item_id → idx | `artifacts/gsasrec/item_id_to_idx.pkl` | 276,305 items после `min_pop≥5` |
| Кэш per-user скоров | `artifacts/user_scores_cache/scores.parquet` | топ-K (user_id, item_id, score, rank) |
| Data infra | `src/data/{yambda_loader,splits}.py` | загрузка + GTS |
| Скорер | `src/scorer/{gsasrec,gbce_loss,train,inference}.py` | замороженный, не трогаем |

**Что НЕ сделано в Phase 1 и нужно в Phase 2:**
- `src/data/group_synthesis.py` — заготовка под random-группы (шаг 3).
- Аудиоэмбеддинги (14 GB) — не выгружены; нужен subset на 276k items (≈ 135 MB).

## Compute (обновлено)

Colab переехал с A100 (40 GB VRAM) на **L4 / G4 с 95 GB RAM**. Это снимает риск «subset аудио не влезет в RAM» (135 MB << 95 GB), и позволяет грузить `embeddings.npy` целиком в память без `memmap`. Скорость обучения агрегаторов по сравнению с A100 чуть ниже, но для 4 небольших моделей это некритично.

## Зафиксированные решения

| Решение | Обоснование |
|---|---|
| Per-user скорер заморожен | План из CLAUDE.md §2; все группы видят один и тот же `s_{u,i}` |
| Loss агрегаторов — pairwise BPR + popularity-negatives | CLAUDE.md §4 (`bpr_loss.py`), как в AGREE/GroupIM |
| Первая итерация — только random-группы, размеры 2–5 из литературы | Memory + CLAUDE.md §8 |
| Метрики первой итерации — только NDCG@10/20 | CLAUDE.md §6, H1-минимальная; Jain/Disagreement/DFH позже |
| AlignGroup не включаем | Несовместим с ephemeral+frozen-scorer setup (см. README) |
| Кандидатный пул `C_G` — объединение топ-K каждого члена группы | Чтобы агрегатор не считал по всему каталогу; согласовано с CLAUDE.md §4 |
| Аудио — range-read нужного subset через `HfFileSystem` | Чтобы не качать 14 GB; subset 276k×128 float32 ≈ 135 MB |

## Открытые вопросы (решаем по ходу)

- **Ground truth для group-таргета.** Union по test-listens членов группы / intersection / только likes? Влияет на форму NDCG. Стартуем с **union по listens в test-окне**, на отдельной ячейке считаем intersection-вариант как sanity.
- **K для кэша кандидатов.** В Phase 1 кэш уже сохранён, проверить какой K взят (вероятно 200–500); если мал — пересчитать в шаге L.
- **λ для MI-loss в GroupIM.** Подбираем на val, стартовая сетка {0.1, 0.5, 1.0}.
- **Размер групп в test.** Стратифицируем по size ∈ {2,3,4,5}, равные доли (по 25%).

## Прогресс по шагам

| Шаг | Описание | Артефакт | Статус |
|---|---|---|---|
| 1 | Sanity-чек кэша скоров + фикс архитектурного бага с pad | патч `gsasrec.py` + чистый кэш | ✅ |
| 2 | Subset аудиоэмбеддингов 276k items → `artifacts/audio/embeddings.npy` | numpy + item_id index | 🟡 код готов, ждёт запуска на Colab |
| 3 | `src/data/group_synthesis.py` — random-группы, размер 2–5 | модуль + smoke | ⬜ |
| 4 | Аудиопрофиль пользователя $\bar{a}_u$ (mean по истории listen+) | `artifacts/audio/user_profiles.npy` | ⬜ |
| 5 | `src/eval/metrics.py` (NDCG@K) + `src/eval/group_eval.py` | модули + unit-тест на toy-данных | ⬜ |
| 6 | `src/training/bpr_loss.py` + `src/training/group_trainer.py` | общий цикл | ⬜ |
| 7 | `src/aggregators/base.py` + `agree.py` (ID-based AGREE) | модуль | ⬜ |
| 8 | `src/aggregators/groupim.py` + MI-дискриминатор | модуль | ⬜ |
| 9 | `src/aggregators/audio_agree.py` | модуль | ⬜ |
| 10 | `src/aggregators/group_cross_attn.py` | модуль | ⬜ |
| 11 | `notebooks/04_train_aggregators.ipynb` — обучить все 4 на одном split групп | чекпоинты в `artifacts/aggregators/` | ⬜ |
| 12 | `notebooks/05_eval_groups.ipynb` — NDCG@10/20 на test-группах, бутстрап CI | csv в `artifacts/eval_results/` | ⬜ |
| 13 | `notebooks/06_results_analysis.ipynb` — финальная таблица + графики + LaTeX-фрагмент | заметки + figures | ⬜ |
| 14 | Тривиальные бейзлайны (AVG / LM / MP) как функции при оценке | дописать в `group_eval.py` | ⬜ |

**Критический путь:** 1 → (2, 3 параллельно) → 4 → 5, 6 → 7..10 → 11 → 12 → 13. Шаги 5 и 6 можно делать параллельно с 7..10.

## План по ноутбукам

| Ноутбук | Шаги | Что должно получиться на выходе |
|---|---|---|
| `04_train_aggregators.ipynb` | 11 | 4 чекпоинта; графики train/val loss; val NDCG@10 по эпохам |
| `05_eval_groups.ipynb` | 12, 14 | csv-таблица: метод × NDCG@{10,20} × {bootstrap CI}; срез по размеру группы |
| `06_results_analysis.ipynb` | 13 | финальная таблица + 2-3 figure для ВКР |

## Контракты данных (для согласованности модулей)

- **Group sample (train/val/test):** dict `{group_id: int, members: list[int], candidates: list[int], targets: list[int]}`. Кандидаты = union топ-K членов; targets = union test-listens членов (per ground-truth решение выше).
- **Batch агрегатора:** соответствует сигнатуре `GroupAggregator.forward` из CLAUDE.md §4. `per_user_scores` подгружаются из `scores.parquet` (там, где item ∈ C_G; для отсутствующих ставим `-inf` или 0 — решить в шаге O).
- **Аудиокэш:** `embeddings.npy` shape `[n_items, 128]`, индексируется тем же `item_id_to_idx.pkl`, что и SASRec. Это критично, чтобы не было сдвига индексов.

## Риски и митигации

| Риск | Митигация |
|---|---|
| ~~Subset аудио не помещается в Colab RAM~~ | Снято: G4 даёт 95 GB RAM, грузим целиком |
| Union-таргет в test пустой для маленьких групп | На шаге 5 логировать долю групп с пустым target; при >5% — пересмотреть GT-стратегию |
| GroupIM MI-loss доминирует над BPR | Сетка λ; ранний log mi/bpr ratio на val |
| Полусырая реализация cross-attention медленная на больших C_G | Ограничить C_G размером ≤ 1024 (merge топ-K с урезанием) |
| 9.2k пользователей мало для 4 моделей × eval | Bootstrap CI обязательно; не overclaim различия в ВКР |

## История сессий

### 2026-05-11 — старт Phase 2
- Phase 1 закрыт (см. `PHASE_1_LOG.md`), создан этот журнал.
- Следующий шаг — 1: открыть `scores.parquet`, проверить K, схему, покрытие пользователей.
- Зафиксировано: Colab переехал на L4/G4 с 95 GB RAM (новее A100 по поколению), риск RAM для аудио снят.

### 2026-05-11 — Шаг 1: sanity-чек кэша + фикс архитектурного бага ✅

**Кэш `artifacts/user_scores_cache/scores.parquet`:** 1.83M строк, K=200 на юзера, 9170 uids (из 9194 train; 24 отфильтрованы как <2 events). Схема: `uid` int64, `item_idx` int64, `score` float32, `rank` int32. item_idx ∈ [1, 276299], PAD=0 исключён. K=200 хватает для `C_G = union(top-K members)` при размере групп ≤5.

**Найдено и пофикшено два бага:**

1. **Inference запускался с `exclude_history=True`** — противоречит Phase 1 eval-протоколу (для музыки маскирование занижает NDCG в 3 раза, см. PHASE_1_LOG). Исправлено флагом в ноутбуке.
2. **Архитектурный баг `GSASRec.forward`:** комбинация `causal_mask + src_key_padding_mask` для query-позиций с all-masked keys давала `softmax(-inf) = NaN`, который через residual'ы протекал до позиции 199. Симптом — 1222 юзера (13.3%) с полностью NaN-кэшем (короткая train-история <200 events). Локально воспроизведено на synthetic n_real ∈ {1..199}; n_real=200 работал. **Фикс** в [src/scorer/gsasrec.py](src/scorer/gsasrec.py): заменил `src_key_padding_mask` на per-batch 3D `attn_mask [B*n_heads, L, L]` с causal+pad-key masking, но всегда доступной диагональю (любая query attend на себя минимум). Distribution shift для real-позиций нулевой, для full-200 юзеров выход бит-в-бит идентичен старому.

**Латентный эффект на Phase 1:** `evaluate_ndcg` тоже страдал → короткие юзеры давали 0 contribution. Baseline 0.0726 слегка занижен; re-eval не делаю, для текста ВКР — footnote.

**Финальный кэш:** NaN: 0 (было 244,400), 52,642 уникальных item_idx в union топ-200 (+2.2k vs до фикса), score: mean 4.70, range [-0.22, 17.26], duplicates 0.

**Открытое:** возможно поднять K до 500 для popularity-negatives — решу на шаге 6.

### 2026-05-11 — Шаг 2: subset аудиоэмбеддингов 🟡 код готов

**Подход.** Качаем `embeddings.parquet` (14 GB, 7.72M × 128) целиком через `hf_hub_download` на Colab-диск, читаем две колонки в `pyarrow.Table`, фильтруем `np.isin(item_id, target_ids)` по 276,305 items из Phase 1, сохраняем `[n_items+1, 128]` float32 (~135 MB), row 0 — PAD. Локально файл не оседает — результат скачиваем после Colab-прогона.

**Артефакты:**
- [src/data/audio_embeddings.py](src/data/audio_embeddings.py) — `extract_audio_subset(item_id_to_idx, output_path, use_normalized=False)`.
- [notebooks/04_audio_subset.ipynb](notebooks/04_audio_subset.ipynb) — 4 ячейки: bootstrap → загрузка `item_id_to_idx` → вызов функции → sanity-check.

**Probe схемы parquet:** `num_row_groups=30`, `num_rows=7,721,749`. Колонки: `item_id uint32`, `embed large_list<double>`, `normalized_embed large_list<double>` (dim 128). Берём `embed`, нормированный вариант — флагом при необходимости.

**Не запущено.** Ждём Colab-сессию.

**Открытое.** Если `seen`-чек покажет пропуски — значит, в Phase 1 после `min_pop≥5` остались id, которых нет в каталоге эмбеддингов; обработаем при первом запуске.
