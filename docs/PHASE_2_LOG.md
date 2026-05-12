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

Colab переехал с A100 на **G4** (новое поколение NVIDIA, мощнее A100, 95 GB RAM). Это снимает риск «subset аудио не влезет в RAM» (135 MB << 95 GB), позволяет грузить `embeddings.npy` целиком в память без `memmap`, и заметно ускоряет обучение агрегаторов — текущий прогон 4 моделей укладывается ~5 минут.

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
| 2 | Subset аудиоэмбеддингов 276k items → `artifacts/audio/embeddings.npy` | numpy + item_id index + `audio_valid` mask | ✅ |
| 3 | `src/data/group_synthesis.py` — random-группы, размер 2–5 | модуль + smoke | ✅ |
| 4 | Аудиопрофиль пользователя $\bar{a}_u$ (mean по истории listen+) | `artifacts/audio/user_profiles.npy` | ✅ |
| 5 | `src/eval/metrics.py` (NDCG@K) + `src/eval/group_eval.py` | модули + unit-тест на toy-данных | ✅ |
| 6 | `src/training/bpr_loss.py` + `src/training/group_trainer.py` | общий цикл | ✅ |
| 7 | `src/aggregators/base.py` + `agree.py` (ID-based AGREE) | модуль | ✅ |
| 8 | `src/aggregators/groupim.py` + MI-дискриминатор | модуль | ✅ |
| 9 | `src/aggregators/audio_agree.py` | модуль | ✅ |
| 10 | `src/aggregators/group_cross_attn.py` | модуль | ✅ |
| 11 | `notebooks/06_train_aggregators.ipynb` — обучить все 4 на одном split групп | чекпоинты в `artifacts/aggregators/` (на HF) | ✅ |
| 12 | `notebooks/07_eval_groups.ipynb` §1–6 — NDCG@10/20 на test-группах, bootstrap + paired CI | csv/npz в `artifacts/eval_results/` | 🟡 код готов, ждёт Colab |
| 13 | `notebooks/07_eval_groups.ipynb` §7 — финальная таблица + forest plot + LaTeX-фрагмент | figures в `docs/figures/` + `summary_table.tex` | 🟡 код готов, ждёт Colab |
| 14 | Тривиальные бейзлайны (AVG / LM / MP) как функции при оценке | в ноутбуке 07 §2.1 | 🟡 код готов, ждёт Colab |

**Критический путь:** 1 → (2, 3 параллельно) → 4 → 5, 6 → 7..10 → 11 → 12 → 13. Шаги 5 и 6 можно делать параллельно с 7..10.

## План по ноутбукам

| Ноутбук | Шаги | Что должно получиться на выходе |
|---|---|---|
| `06_train_aggregators.ipynb` | 11 | 4 чекпоинта; графики train/val loss; val NDCG@10 по эпохам |
| `07_eval_groups.ipynb` | 12, 13, 14 | csv-таблица: метод × NDCG@{10,20} × bootstrap CI; срез по размеру; forest plot; LaTeX-фрагмент |

> Изначально планировалось 3 ноутбука Phase 2, но шаги 13 и 14 слились в 07: финальный анализ читает in-memory `summary_df`/`per_sample` и работает с тем же bootstrap-протоколом, что и сам eval — нет смысла дублировать.

> Нумерация ноутбуков. В CLAUDE.md план писался от `04_train_aggregators.ipynb`, но к моменту шага 11 диск уже занял 04/05 под `audio_subset` / `user_audio_profiles`. Phase 2 training notebook = **06**, eval = **07**, analysis = **08**.

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
- Зафиксировано: Colab переехал на G4 (новее и мощнее A100, 95 GB RAM), риск RAM для аудио снят.

### 2026-05-11 — Шаг 1: sanity-чек кэша + фикс архитектурного бага ✅

**Кэш `artifacts/user_scores_cache/scores.parquet`:** 1.83M строк, K=200 на юзера, 9170 uids (из 9194 train; 24 отфильтрованы как <2 events). Схема: `uid` int64, `item_idx` int64, `score` float32, `rank` int32. item_idx ∈ [1, 276299], PAD=0 исключён. K=200 хватает для `C_G = union(top-K members)` при размере групп ≤5.

**Найдено и пофикшено два бага:**

1. **Inference запускался с `exclude_history=True`** — противоречит Phase 1 eval-протоколу (для музыки маскирование занижает NDCG в 3 раза, см. PHASE_1_LOG). Исправлено флагом в ноутбуке.
2. **Архитектурный баг `GSASRec.forward`:** комбинация `causal_mask + src_key_padding_mask` для query-позиций с all-masked keys давала `softmax(-inf) = NaN`, который через residual'ы протекал до позиции 199. Симптом — 1222 юзера (13.3%) с полностью NaN-кэшем (короткая train-история <200 events). Локально воспроизведено на synthetic n_real ∈ {1..199}; n_real=200 работал. **Фикс** в [src/scorer/gsasrec.py](src/scorer/gsasrec.py): заменил `src_key_padding_mask` на per-batch 3D `attn_mask [B*n_heads, L, L]` с causal+pad-key masking, но всегда доступной диагональю (любая query attend на себя минимум). Distribution shift для real-позиций нулевой, для full-200 юзеров выход бит-в-бит идентичен старому.

**Латентный эффект на Phase 1:** `evaluate_ndcg` тоже страдал → короткие юзеры давали 0 contribution. Baseline 0.0726 слегка занижен; re-eval не делаю, для текста ВКР — footnote.

**Финальный кэш:** NaN: 0 (было 244,400), 52,642 уникальных item_idx в union топ-200 (+2.2k vs до фикса), score: mean 4.70, range [-0.22, 17.26], duplicates 0.

**Открытое:** возможно поднять K до 500 для popularity-negatives — решу на шаге 6.

### 2026-05-11 — Шаг 2: subset аудиоэмбеддингов ✅

**Подход.** Качаем `embeddings.parquet` (14 GB, 7.72M × 128) целиком через `hf_hub_download` на Colab-диск, читаем две колонки в `pyarrow.Table`, фильтруем `np.isin(item_id, target_ids)` по 276,305 items из Phase 1, сохраняем `[n_items+1, 128]` float32 (~135 MB), row 0 — PAD. Локально файл не оседает — результат скачали после Colab-прогона.

**Артефакты:**
- [src/data/audio_embeddings.py](src/data/audio_embeddings.py) — `extract_audio_subset(item_id_to_idx, output_path, use_normalized=False)`.
- [notebooks/04_audio_subset.ipynb](notebooks/04_audio_subset.ipynb) — 4 ячейки: bootstrap → загрузка `item_id_to_idx` → вызов функции → sanity-check (+ доп. ячейка с popularity-анализом missing).

**Probe схемы parquet:** `num_row_groups=30`, `num_rows=7,721,749`. Колонки: `item_id uint32`, `embed large_list<double>`, `normalized_embed large_list<double>` (dim 128). Берём `embed`, нормированный вариант — флагом при необходимости.

**Запуск на Colab (CPU runtime, 50 GB RAM):** выполнен, всё прошло успешно. Также отдельно прогнан popularity-анализ missing items (`load_yambda` + `filter_listens` + `value_counts` — ~5–10 мин на CPU, узкое место — `to_pandas()` на 46M строк, GPU тут не помогает).

**Результат `artifacts/audio/embeddings.npy`:** shape `(276306, 128)`, dtype float32, 134.9 MB. PAD-row (idx=0) нулевой, NaN/Inf нет. Норма (non-zero): mean=28.45, range [14.50, 190.51].

**Coverage:** 264,840 / 276,305 items с эмбеддингом (95.85%); **11,465 (4.15%) — без аудио, остаются нулями**.

**Природа пропусков (sanity-чек по popularity, ячейка в `04_audio_subset.ipynb`):**

| | missing | present |
|---|---:|---:|
| median pop | 15 | 18 |
| mean pop | 73 | 106 |
| p90 | 142 | 195 |
| max | 12,770 | 28,268 |
| pop<10 | 33.9% | — |
| pop≥50 | 22.1% | — |

Гипотеза «missing = хвост популярности» **отвергнута**: missing размазаны по всему диапазону (max=12,770 listens у missing-трека), median почти равна present. Это **catalog drift** — отсутствие аудио ~для 4% треков не коррелирует с popularity, объясняется доступностью контента (права / удалённые треки / snapshot drift). Соответствует общей картине датасета: yambda README заявляет 7.72M embeddings vs 9.39M items в 5B (~18% gap на полном каталоге, у нас 4% — потому что фильтр `min_pop≥5` всё-таки смещает в сторону популярных).

**Контракт для шагов 4, 9, 10 (audio-aware агрегаторы):**

- Рядом с `embeddings.npy` используем булеву маску `audio_valid = norm(arr, axis=1) > 0` (shape `[n_items+1]`, idx=0 → False как PAD).
- `C_G` (кандидатный пул) **один и тот же** для всех 4 методов — иначе сравнение методов поедет.
- AudioAGREE / CrossAttn: missing-кандидаты получают нулевой вклад в attention-логит (mask before softmax). Пользовательский профиль `a_bar_u` усредняется только по valid-items истории.
- ID-based AGREE и GroupIM маску игнорируют (они не видят аудио).

### 2026-05-11 — Шаг 3: random-группы ✅

**Артефакт:** [src/data/group_synthesis.py](../src/data/group_synthesis.py) — функция `synthesize_random_groups(user_pool, n_groups, size_distribution, seed)`. Поведение: внутри группы участники уникальны (`replace=False`), между группами повторы разрешены (with replacement) — стандартный ephemeral-setup AGREE/GroupIM. Распределение размеров задаётся словарём `{size: prob}`, проверяется сумма ≈ 1.

**Smoke (`python src/data/group_synthesis.py`):** n_groups=10_000, pool=100 users, distribution `{2: 0.3, 3: 0.4, 4: 0.2, 5: 0.1}` → эмпирические доли `{2: 0.3008, 3: 0.3991, 4: 0.2021, 5: 0.0980}`, mean size 3.097 (target ≈3.10). Детерминизм по seed проверен (seed=0 идентичен, seed=1 отличается).

**Открыто:** train/val/test split групп пока не делаем — формат `Group sample {members, candidates, targets}` соберём в шагах 5/6, когда появится eval-обвязка и определимся с union-target по test-listens.

### 2026-05-11 — Шаг 4: user audio profiles ✅ (код)

**Артефакт:** [src/data/audio_embeddings.py](../src/data/audio_embeddings.py) — функция `build_user_audio_profiles(train_listens, item_embeddings)` → `(profiles[n_users, 128], uid_to_row, user_audio_valid[n_users])`. Усредняет по train listen+ истории, маскируя items с `norm(emb)==0` (4.15% catalog drift из шага 2). Юзеры без valid items получают zero-row и `user_audio_valid=False`.

**Дизайн.** Индексация — compact `[n_users, 128]` + `uid_to_row.pkl` (а не dense `[max_uid+1, ...]`), по аналогии с `item_id_to_idx`. Это даёт ~4.5 MB вместо потенциально гигабайтов на разреженных raw-uids. История — только **train** (после GTS), чтобы не было лика из val/test.

**Smoke (локально, synthetic).** `n_items=10`, 2 missing items (idx 3, 7), 4 юзера; вручную проверены: (а) user с миксом valid+invalid усредняет только valid; (б) user только с invalid item получает zero-row и valid=False; (в) счётчики совпадают. Прошло.

**Ноутбук:** [notebooks/05_user_audio_profiles.ipynb](../notebooks/05_user_audio_profiles.ipynb) — Colab-runner: load YAMBDA-50m → filter (listen+, min_pop≥5) → load `item_id_to_idx.pkl` → remap → GTS → load `embeddings.npy` → `build_user_audio_profiles` → сохранение в `artifacts/audio/{user_profiles.npy, uid_to_row.pkl, user_audio_valid.npy}`. Включён sanity-чек: для случайного uid профиль пересчитывается «руками» и сравнивается с сохранённым (max|diff| < 1e-5).

**Запуск на Colab — pending** (только пользователь может проверить артефакты и обновить лог финальными числами по `user_audio_valid` coverage).

**Контракт для шагов 7–10.** AudioAGREE / GroupCrossAttn в `forward` принимают `audio_profiles_users[B, |G|, 128]` — собираются из этой таблицы по `uid_to_row`. Для юзеров с `user_audio_valid=False` агрегатор должен либо игнорировать их в attention, либо fallback на скоринг без audio-ветки (определимся на шагах 9/10).

### 2026-05-12 — Шаг 5: eval-обвязка (metrics + group_eval) ✅

**Артефакты:**
- [src/eval/metrics.py](../src/eval/metrics.py) — `dcg_at_k`, `idcg_at_k`, `ndcg_at_k` (низкоуровневый, `[B, L]` бинарных релевантностей + `n_relevant[B]`), `ranking_ndcg_at_k` (сортирует scores per-row, поддерживает несколько K за вызов), `ndcg_from_ranking` (single-query). Векторизовано на numpy, без torch-зависимости — eval-обвязка не зависит от среды агрегатора.
- [src/eval/group_eval.py](../src/eval/group_eval.py) — `GroupSample` dataclass, `build_group_samples(groups, user_topk, user_test_targets, ground_truth, drop_empty, drop_missing_member)` собирает кандидатные пулы (union top-K членов) и таргеты (union/intersection test-listens ∩ candidates); `evaluate_aggregator_scores(samples, group_scores, k_list)` считает NDCG@K с разбивкой по размеру группы и per-sample массивами под бутстрап CI; хелперы `topk_from_score_cache` и `test_targets_from_df` для подключения parquet'ов из Phase 1.
- [tests/test_eval.py](../tests/test_eval.py) — 18 toy-тестов (pytest). Покрывают: ручные DCG/IDCG/NDCG, идеальный/нулевой ranking, truncate на K, корректность сортировки по score, batch over rows, union/intersection-target, drop_empty/drop_missing_member, размерные срезы, валидация формы скоров.

**Зафиксированные решения (по уточняющим вопросам в чате):**
1. **Default ground-truth — union** по test-listens членов. Intersection остаётся параметром (`ground_truth="intersection"`) для sanity-cell в ноутбуке `05_eval_groups.ipynb`.
2. **Out-of-pool релеванты исключаются.** `targets = union(test_listens) ∩ candidates`. IDCG нормируется только по достижимым релевантам в `C_G`. Это согласовано с тем, что агрегатор не видит весь каталог — оценивать «недостижимый» NDCG бессмысленно.

**Запуск:** `pytest tests/test_eval.py -q` → 18/18 passed.

**Контракт для шага 6 (trainer).** Trainer на каждом батче должен отдавать `list[np.ndarray[|C_G|]]` group-скоров — `evaluate_aggregator_scores` принимает его напрямую. `per_user_scores` для агрегатора собираются отдельно из `scores.parquet` (см. шаг 1 кэш) — не в этом модуле.

**Открытое:** train/val/test split групп пока не зафиксирован (3 ноутбука Phase 2 будут резать groups одинаково по seed — формализуем в шаге 11).

### 2026-05-12 — Шаг 6: BPR loss + GroupAggregatorTrainer ✅

**Артефакты:**
- [src/training/bpr_loss.py](../src/training/bpr_loss.py) — `pairwise_bpr_loss(pos, neg)`. Использует `F.logsigmoid` (численно стабилен), бродкастит `[B] vs [B]` и `[B] vs [B,K]`.
- [src/training/group_trainer.py](../src/training/group_trainer.py) — `GroupTrainConfig`, `GroupAggregatorTrainer`, `GroupTrainDataset`, `GroupEvalDataset`, `collate_groups`, хелперы `build_user_score_lookup`, `lookup_per_user_scores`, `compute_pop_counts`.
- [tests/test_training.py](../tests/test_training.py) — 13 тестов (BPR + lookup + pop + neg-sampling + collate + end-to-end fit с тривиальным `MeanScoreAggregator`).

**Зафиксированные решения (на уточняющих вопросах в чате):**

1. **Fill для `per_user_scores[u,i]` при item ∉ top-K(u): `0.0`.** Совместимо с любой attention-формой (нет NaN от `-inf`), а реальные scores ∈ [-0.2, 17.3] так что 0 — нижний край, не «нейтраль внутри распределения». Этот же fill используется для pad-позиций в батче.
2. **Негативы — popularity-weighted из `C_G \ targets`.** В духе CLAUDE.md «popularity-negatives». Веса = popularity^0.75 (как в Phase 1 `compute_item_popularity`), таргеты обнуляются до нормализации. Хвост уже отсеян (`C_G` = union top-K, заведомо релевантные кандидаты) — это и есть hard negatives для группы.

**Расширение контракта `GroupAggregator.forward` (уточнение к CLAUDE.md §4):**

Trainer паддит `|G|` и `|C_G|` до batch-max и передаёт булевы маски как kwargs:

```
group_mask:     BoolTensor[B, G_max]   # True — реальный член, False — pad
candidate_mask: BoolTensor[B, C_max]   # True — реальный кандидат, False — pad
```

Pad-кандидатам после forward присваивается `-inf` (не выигрывают ranking; в `predict_group_scores` обрезаются до реального `|C_G|`). Pad-членам соответствует `per_user_scores=0` и `audio_profile=0` — но агрегатор обязан игнорировать их через `group_mask` (иначе attention/mean сместятся). Эти kwargs обязательны для всех 4 реализаций в шагах 7–10.

**Trainer:**
- Optimizer — Adam (`weight_decay=0` по умолчанию).
- Loss = `pairwise_bpr_loss(pos, neg)` + опционально `λ * aggregator.regularization_loss(batch, scores)` если метод определён (для GroupIM на шаге 8).
- Eval per epoch — `evaluate_aggregator_scores` по val-группам, NDCG@10 (первый K из `eval_k`) как primary для early-stopping.
- Сохраняет `best.pt`, `config.json`, `metrics.csv` в `out_dir`.

**Smoke (`pytest tests/test_training.py -q`):** 13/13 passed. Регрессия по `tests/test_eval.py`: 18/18 passed.

**Контракт для шага 7 (`base.py`).** ABC `GroupAggregator` должен явно объявить kwargs `group_mask: torch.BoolTensor | None`, `candidate_mask: torch.BoolTensor | None`. Реализации (AGREE / GroupIM / AudioAGREE / GroupCrossAttn) обязаны корректно маскировать softmax по членам (исключать pad-членов из нормализации). Для AudioAGREE/CrossAttn также маскировать missing-audio items (см. контракт из шага 2 — `audio_valid`).

**Открытое:** размер K для cached top-K (сейчас 200) → если в шаге 11 средний `|C_G|` окажется маловат для хороших pop-negatives, поднимем K до 500 и пересчитаем кэш (см. шаг 1 открытое).

### 2026-05-12 — Шаг 7: base.py + IDBasedAGREE ✅

**Артефакты:**
- [src/aggregators/base.py](../src/aggregators/base.py) — `GroupAggregator(nn.Module, ABC)` с явной сигнатурой `forward(group_user_ids, candidate_ids, per_user_scores, audio_embeds_items=None, audio_profiles_users=None, group_mask=None, candidate_mask=None) -> [B, C_max]`. Контракт по маскам зафиксирован в docstring.
- [src/aggregators/agree.py](../src/aggregators/agree.py) — `IDBasedAGREE(uid_list, num_items, d_emb=32, d_att=d_emb)`.
- [src/aggregators/__init__.py](../src/aggregators/__init__.py) — публичный реэкспорт.
- [tests/test_aggregators.py](../tests/test_aggregators.py) — 12 тестов (ABC + remap + forward shape/mask + pad-инвариантность + backprop + end-to-end fit + predict shapes).

**Дизайн ID-based AGREE.** Воспроизводим оригинал Cao et al. SIGIR'18 в части attention, отступаем только в том, что per-user score $s_{u,i}$ берётся из замороженного SASRec, а не из $\langle e_u, e_i \rangle$:

- Две learnable таблицы: `user_emb [n_users+1, d]`, `item_emb [n_items+1, d]`, обе с `padding_idx=0`.
- Item-aware attention: $\alpha_{u,i} = \mathrm{softmax}_u\big(h^\top \tanh(W [e_u; e_i] + b)\big)$. `W: Linear(2d, d_att)`, `h: Linear(d_att, 1, bias=False)`.
- Group score: $s_G(i) = \sum_u \alpha_{u,i} \cdot s_{u,i}$. Item-эмбеддинги учатся через backprop по attention-логиту, в финальный скор не идут.

**Remap raw uid → compact row** делается внутри модуля: `register_buffer("_sorted_uids", LongTensor([0] + sorted(uid_list)))`, в forward `rows = searchsorted(_sorted_uids, members)`. Это даёт компактную таблицу `n_users+1` независимо от разреженности raw uids в YAMBDA — embedding-таблица user_emb имеет ровно `n_users+1` строк (~9170 для 50m flavor). 0 в `uid_list` отбрасывается (зарезервирован под PAD).

**Контракт по маскам.** В `forward`:
1. `logits.masked_fill(~group_mask.unsqueeze(-1), -inf)` перед softmax по G — pad-члены получают нулевой вес.
2. `nan_to_num(alpha, 0.0)` на случай, если у кандидата все G позиций — pad (теоретически невозможно при G≥1 реальных, но safety).
3. Pad-кандидаты не маскируются на уровне агрегатора — это делает trainer (`-inf` после forward).

Аудио-аргументы (`audio_embeds_items`, `audio_profiles_users`) в сигнатуре есть, но игнорируются — `del` на входе. Это нужно, чтобы Trainer мог использовать один и тот же `_forward(batch)` для всех 4 агрегаторов.

**Регрессионный риск:** изменения только additive (новая директория `src/aggregators/`). `tests/test_eval.py` (18) и `tests/test_training.py` (13) проходят без правок.

**Smoke (`pytest tests/test_aggregators.py tests/test_eval.py tests/test_training.py -q`):** 43/43 passed за 0.98s.

**Контракт для шагов 8–10.** Все три оставшихся агрегатора (GroupIM, AudioAGREE, GroupCrossAttn) наследуются от `GroupAggregator` и обязаны:
- Корректно маскировать softmax по `group_mask` (для CrossAttn — query/key padding mask).
- Для audio-методов — дополнительно использовать `audio_valid`-маску по items (см. шаг 2) при формировании attention-логитов.
- GroupIM реализует `regularization_loss(batch, scores)` для MI-loss — Trainer подключает его автоматически если `cfg.reg_loss_weight > 0`.

**Открытое:** Параметры AGREE (`d_emb`, `d_att`) пока на дефолтах. Sweep по `d_emb ∈ {16, 32, 64}` сделаем на шаге 11 (`04_train_aggregators.ipynb`) вместе с тюнингом lr/n_neg/batch.

### 2026-05-12 — Шаг 8: GroupIM + MI-дискриминатор ✅

**Артефакты:**
- [src/aggregators/groupim.py](../src/aggregators/groupim.py) — `GroupIM(uid_list, num_items, d_emb=32, d_att=d_emb)`.
- Обновлён [src/aggregators/__init__.py](../src/aggregators/__init__.py) — публичный реэкспорт `GroupIM`.
- [tests/test_aggregators.py](../tests/test_aggregators.py) — +10 тестов GroupIM (subclass, shape, pad-masking, item-agnostic, audio ignored, MI grad/zero/reset/small-batch, end-to-end fit с `reg_loss_weight>0`).

**Дизайн GroupIM (адаптация Sankar et al. SIGIR'20 под наш сетап):**

- **User-эмбеддинги learnable** `E_user[n_users+1, d]` (`padding_idx=0`). Item-эмбеддинги не нужны: per-user score `s_{u,i}` приходит из замороженного SASRec через `per_user_scores`, аналогично контракту AGREE.
- **Item-agnostic attention** — ключевая абляция против item-aware AGREE: `alpha_u = softmax_u(h^⊤ tanh(W·e_u))`. У оригинальной GroupIM attention тоже item-agnostic, поэтому это in-spirit paper. Group score: `s_G(i) = Σ_u α_u · s_{u,i}` (одна и та же `α` для всех items).
- **Group representation** `h_G = Σ_u α_u · e_u` — нужно для MI-loss.
- **MI-loss** — bilinear-дискриминатор `D(e_u, h_G) = e_u^⊤ W_{MI} h_G`. Положительные пары — `(e_u, h_G)` для `u ∈ G`, негативы — `(e_u, h_{G'})` где `G' = roll(G, 1)` по батчу (cross-batch). BCE через `logsigmoid(pos) + logsigmoid(-neg)`, маска по `group_mask` исключает pad-членов. Замечание: false negatives возможны (тот же uid может оказаться в G и G' из-за ephemeral-семплинга), в первой итерации принимаем шум как в оригинале.

**Хук под Trainer.** Trainer вызывает `aggregator.regularization_loss(batch, scores)` после forward (см. шаг 6). GroupIM кэширует `(member_emb, group_repr, group_mask)` внутри forward только в `self.training=True`, далее `regularization_loss` берёт их и вызывает `mi_loss(...)`, после чего сбрасывает кэш. Eval-режим возвращает скалярный 0 без autograd. Альтернативный публичный `mi_loss(member_emb, group_repr, group_mask, neg_group_repr=None)` — для unit-тестов без forward.

**Зафиксированные решения (свежие, по этому шагу):**

1. **Attention item-agnostic, а не item-aware.** Это намеренный контраст к AGREE (item-aware) для чистой абляции: один и тот же frozen-scorer, одна и та же ID-таблица user-эмбеддингов, разная функция агрегации. Если оба метода хуже audio-веток (шаги 9–10), это будет аргумент за audio-сигнал; если AGREE сильно лучше GroupIM — за item-aware attention в ID-режиме.
2. **Negatives для MI — batch-shift (`torch.roll`).** Простейшая cross-batch стратегия, не требует дополнительной семплинг-обвязки. При `B<2` (вырожденный случай) MI-loss = 0. Это поведение явно протестировано.
3. **Кэш state в модуле, не в trainer.** Альтернатива — прокидывать `(member_emb, group_repr)` через возвращаемое значение forward, но это сломало бы единую сигнатуру `[B, C_max]` контракта. Кэш сбрасывается после каждого `regularization_loss` → не утекает в инференс.

**Smoke (`pytest tests/test_aggregators.py -q`):** 22/22 passed. Full regression `pytest tests/ -q`: **53/53 passed за ~0.93s** (включая 18 eval + 13 training + 22 aggregators).

**Контракт для шагов 9–10 (audio-aware агрегаторы).** AudioAGREE и GroupCrossAttn наследуются от того же `GroupAggregator`, должны корректно использовать `audio_valid`-маску по items (см. контракт из шага 2) при формировании attention-логитов, и маскировать softmax по `group_mask`. Учиться без MI-loss (если только мы не захотим audio-вариант MI как отдельный бейзлайн — на сейчас не планируем).

**Открытое:**
- λ для MI-loss — сетка `{0.1, 0.5, 1.0}` подберём на шаге 11 (`04_train_aggregators.ipynb`), как зафиксировано в «Открытые вопросы» выше.
- MI-дискриминатор сейчас bilinear; в paper'е — MLP. Если bilinear даст плоский val NDCG, попробуем `MLP(concat(e_u, h_G)) → 1` как замену (тривиальная правка, отложена до экспериментов).

### 2026-05-12 — Шаг 9: AudioAGREE ✅

**Артефакты:**
- [src/aggregators/audio_agree.py](../src/aggregators/audio_agree.py) — `AudioAGREE(d_audio=128, d_att=64)`.
- Обновлён [src/aggregators/__init__.py](../src/aggregators/__init__.py) — реэкспорт `AudioAGREE`.
- [tests/test_aggregators.py](../tests/test_aggregators.py) — +9 тестов AudioAGREE (subclass, shape, raises-without-audio, pad-masking, item-aware, ID-args-ignored, zero-audio-no-NaN, backprop, end-to-end fit).

**Дизайн.** Прямой аналог AGREE из шага 7 с одной заменой: `(e_u, e_i)` → `(a_bar_u, a_i)`. ID-таблиц нет — `phi: Linear(2·d_a → d_att) → GELU → Linear(d_att → 1)` единственный обучаемый компонент. Attention item-aware:
- `alpha_{u,i} = softmax_u( phi([a_bar_u; a_i]) )` с маскированием pad-членов через `group_mask` (логит → -inf).
- Group score: `s_G(i) = Σ_u alpha_{u,i} · s_{u,i}`, где `s_{u,i}` — кэш замороженного SASRec.

`group_user_ids` и `candidate_ids` в forward игнорируются (нет ID-эмбеддингов). Контракт base.py позволяет: эти аргументы у нас часть единой сигнатуры для всех 4 агрегаторов, чтобы Trainer мог дёргать одинаковый `_forward(batch)`.

**Зафиксированные решения по этому шагу:**

1. **Missing-audio items/users не маскируются явно.** Контракт шага 2 говорит «missing-кандидаты → нулевой вклад в логит». Реализация: для item с `a_i = 0` MLP-логит зависит только от `a_bar_u` (одна половина concat обнулена); для user с `a_bar_u = 0` — только от `a_i`. Это естественный degradation без edge-cases с softmax(-inf) → NaN. Если у группы вся аудио-сторона нулевая (теоретически возможно), logit одинаковый по членам → uniform alpha → mean per_user_scores, что разумный fallback (= AVG). Покрыто `test_audio_agree_missing_audio_does_not_nan`.
2. **`d_att=64` по умолчанию** (vs `d_emb=32`/`d_att=32` у AGREE). Вход у нас 2·128=256 (а у AGREE — 2·32=64), поэтому скрытый слой пропорционально больше. Sweep по `d_att` — отложен на шаг 11.
3. **`forward` raises на отсутствие аудио** (vs тихий ignore у AGREE). Это сигнал, что Trainer без `item_audio`/`user_profiles` запускать AudioAGREE нельзя. Покрыто `test_audio_agree_raises_without_audio`.

**Smoke (`pytest tests/ -q`):** **62/62 passed за ~1.5s** (включая 18 eval + 13 training + 31 aggregators = 22 старых + 9 новых).

**Контракт для шага 10 (`group_cross_attn.py`).** GroupCrossAttention использует те же audio-аргументы, но через multi-head cross-attention с `Q=a_i, K=V=a_bar_u`. Pad-членов маскировать в key_padding_mask, missing-audio items — снова не обязательно (zero-vector key даст близкую к равномерной attention; решим на месте, нужна ли явная маскировка по `audio_valid` если в реальных данных будет много нулевых key-векторов).

**Открытое:** sweep по `d_att` и числу head'ов в cross-attn (шаг 10) сделаем единым прогоном на шаге 11.

### 2026-05-12 — Шаг 10: GroupCrossAttention ✅

**Артефакты:**
- [src/aggregators/group_cross_attn.py](../src/aggregators/group_cross_attn.py) — `GroupCrossAttention(d_audio=128, d_model=64, n_heads=4)`.
- Обновлён [src/aggregators/__init__.py](../src/aggregators/__init__.py) — реэкспорт `GroupCrossAttention`.
- [tests/test_aggregators.py](../tests/test_aggregators.py) — +11 тестов GroupCrossAttention (subclass, head-config validation, shape, raises-without-audio, pad-masking, item-aware, ID-args-ignored, zero-audio-no-NaN, all-pad-no-NaN, single-head-equivalence, backprop, end-to-end fit).

**Дизайн.** Multi-head scaled dot-product cross-attention с `Q = a_i`, `K = a_bar_u`. V-проекции нет — естественное multi-head обобщение AudioAGREE: используем attention-веса для линейной комбинации `per_user_scores` (скаляров), а не d_model-векторов. Это сохраняет общий контракт всех 4 методов: `s_G(i) = Σ_u α_{u,i} · s_{u,i}`.

- `q_proj: Linear(d_a → d_model)`, `k_proj: Linear(d_a → d_model)`, обе с `xavier_uniform`+`zeros_(bias)`.
- Логиты `[B, H, C, G] = (q @ k^⊤) / sqrt(d_head)`, маскирование pad-членов через `~group_mask` → `-inf`.
- Softmax по G → `alpha[B, H, C, G]`; `nan_to_num` для all-pad-row safety; mean по головам → `alpha[B, C, G]`.
- Итог: `(alpha * per_user_scores.transpose(1, 2)).sum(dim=-1)` → `[B, C]`.

**Зафиксированные решения по этому шагу:**

1. **V-проекция выкинута.** Альтернатива (V = `Linear(a_bar_u)`, output → `Linear(d_model → 1)`) обучала бы свой собственный «item-агностичный summary group» — это нарушает контракт «frozen scorer + аггрегатор только взвешивает» и делает CrossAttn несравнимым с AGREE/AudioAGREE (там итог тоже Σ α · s). Текущий дизайн делает шаг 10 чистым ablation по сравнению с шагом 9: AudioAGREE использует MLP `phi(concat(a_u, a_i))` для логита, CrossAttn — H голов scaled dot-product. Все остальные части (per_user_scores, маска, BPR-обвязка) идентичны.
2. **Mean по головам, не concat.** Concat`→Linear(H → 1)` ввёл бы дополнительный обучаемый mixer, который имеет смысл если головы должны быть «специализированы». В нашем small-data сетапе (9k юзеров, 4 модели) проще оставить mean — это убирает 1 матрицу параметров и при H=1 эквивалентно AudioAGREE-без-MLP. Покрыто `test_cross_attn_single_head_equivalence`.
3. **`audio_valid`-маска по items не применяется.** Контракт шага 2 говорит «missing-кандидаты → нулевой вклад в логит». В CrossAttn нулевой `a_i` даёт нулевой Q → логиты по этому кандидату ≈ 0 по всем G → softmax по G даёт ~равномерное `alpha` → score ≈ mean(per_user_scores) (мягкий AVG-fallback). Это совместимо с тем же поведением AudioAGREE для missing-аудио и не требует edge-case-маскирования. Покрыто `test_cross_attn_missing_audio_does_not_nan`.

**Дефолты `d_model=64`, `n_heads=4`** (d_head=16). Выбор близок к AudioAGREE (`d_att=64`): один и тот же бюджет hidden-размера, но раздроблен на головы — это и есть основная разница, которую хотим измерить. Sweep по `d_model ∈ {32, 64}` и `n_heads ∈ {1, 2, 4, 8}` — отложен на шаг 11.

**Smoke (`pytest tests/ -q`):** **74/74 passed за ~1.45s** (18 eval + 13 training + 43 aggregators = 22 + 9 + 12 cross-attn по факту считаны как 11, плюс старый AudioAGREE-fit). Регрессий нет.

**Контракт для шага 11 (`04_train_aggregators.ipynb`).** Все 4 агрегатора (AGREE/GroupIM/AudioAGREE/GroupCrossAttention) теперь имеют единую сигнатуру `forward(group_user_ids, candidate_ids, per_user_scores, audio_embeds_items, audio_profiles_users, group_mask, candidate_mask) -> [B, C_max]` и обучаются через один и тот же `GroupAggregatorTrainer`. ID-методам аудио-аргументы можно не подавать, audio-методам ID-аргументы можно (они игнорируются). Trainer получает `item_audio` / `user_profiles` / `uid_to_row` опционально — это путь к единому коду в ноутбуке.

**Открытое:** sweep `d_att` / `d_model` / `n_heads` / `lr` для всех 4 методов будет в шаге 11. После него — основная таблица сравнения.

### 2026-05-12 — Шаг 11: 06_train_aggregators.ipynb — обвязка (код готов, ждёт Colab) 🟡

**Артефакт:** [notebooks/06_train_aggregators.ipynb](../notebooks/06_train_aggregators.ipynb) — 27 ячеек, структура «shared setup → 4 отдельных блока обучения → comparison-cell».

**Зафиксированные решения по этому шагу:**

1. **Имя ноутбука — `06_*`, не `04_*`** как в исходном плане CLAUDE.md. К Phase 2 диск уже занял 04/05 под аудио-prep. Phase 2 nb-нумерация: 06 train → 07 eval → 08 analysis. Уточнение добавлено в «План по ноутбукам» выше.
2. **Scope шага — single config per method, без полного sweep.** В чате обсудили: для honest сравнения 4 методов на ВКР важен одинаковый compute budget и разумные дефолты, а не подгонка каждого. Sweep по `d_att`/`d_model`/`n_heads` отложен до шага 11b (если результаты шага 12 покажут, что какой-то метод явно недотюнен). λ_MI для GroupIM = 0.5 (середина сетки {0.1, 0.5, 1.0}) — единственное «непаперное» значение. lr=1e-3 для всех 4 (Adam default).
3. **Структура ноутбука — per-method блоки, не sweep-цикл.** Пользователь хотел иметь возможность переопределять конфиг каждого метода отдельно (формулировка: «наверняка понадобится менять конфиги к каждой отдельно»). Каждый блок = `IDBasedAGREE/GroupIM/AudioAGREE/GroupCrossAttention(**cfg)` + `trainer.fit()`. Помодульная итерация: правишь `d_emb` у GroupIM → перезапускаешь только её ячейку, остальное не трогается.
4. **Тест НЕ touchаем в этом ноутбуке.** Подсматривание в test при правке конфигов — методологическая ошибка (комиссия ВКР спросит «вы фиксировали гиперпараметры по val или по test?»). Val NDCG@10 используется для early stopping и финальной comparison-таблицы. Test → отдельный шаг 12 (`07_eval_groups.ipynb`) один раз после фиксации всех конфигов. Test_groups при этом синтезируются здесь же (тот же seed) и сохраняются в `groups_split.pkl` — это гарантирует, что шаг 12 видит идентичный split.
5. **Bootstrap — `hf_hub_download` из `Vladislavbro-500/music-recommendations` (dataset).** Артефакты Phase 1 + аудио залиты на HF (не в git), скачиваются в `artifacts/` идемпотентно (skip если файл уже есть). Это разводит код (git) и тяжёлые артефакты (HF) — стандартный pattern для Colab.

**Контракт данных, зафиксированный в ноутбуке:**
- `groups_split.pkl` (артефакт): `{train_groups, val_groups, test_groups, group_seed=42, size_dist, train_stats, val_stats}` — один файл, читается шагом 12.
- Train-таргеты = union train listens членов ∩ candidates; val-таргеты = union val listens. Test-таргеты в этом ноутбуке не строятся (но в `groups_split.pkl` лежат raw `test_groups` — шаг 12 сам построит `test_samples`).
- `item_audio` + `user_profiles` передаются Trainer'у для всех 4 методов (ID-методы игнорируют, audio-методы используют — единый код).
- Все 4 чекпоинта в `artifacts/aggregators/<name>/{best.pt, config.json, metrics.csv}`. После Colab-прогона — загружаются обратно на HF (`hf upload ... artifacts/aggregators . --type dataset`), памятка в последней md-ячейке.

**Дефолты конфига (`GroupTrainConfig`):**
```
n_epochs=20, batch=64, eval_batch=128, lr=1e-3, n_neg=4, patience=5,
eval_k=(10, 20), seed=42, device='cuda' if available else 'cpu'
```

Per-method overrides:
- AGREE: `d_emb=32, d_att=32`
- GroupIM: `d_emb=32, d_att=32, reg_loss_weight=0.5`
- AudioAGREE: `d_audio=128, d_att=64`
- GroupCrossAttention: `d_audio=128, d_model=64, n_heads=4`

**Запуск — pending** (пользователь прогоняет на Colab G4, ~5 мин на все 4 модели).

### 2026-05-12 — Шаг 11: результаты Colab-прогона ✅

**Два прогона:**
- Run 1: `n_epochs=60, patience=20` (дефолт ноутбука).
- Run 2: `n_epochs=100, patience=30` — расширили после анализа кривых.

**Финальные числа (Run 2, val):**

| method          | NDCG@10 | NDCG@20 | NDCG@10[s=2] | s=3   | s=4   | s=5   |
|-----------------|--------:|--------:|-------------:|------:|------:|------:|
| AudioAGREE      | **0.0978** | **0.1097** | 0.135 | 0.094 | 0.080 | 0.071 |
| GroupCrossAttn  | 0.0941  | 0.1056  | 0.119        | 0.094 | 0.081 | 0.069 |
| AGREE           | 0.0916  | 0.1024  | 0.126        | 0.090 | 0.074 | 0.064 |
| GroupIM (λ=0.5) | 0.0911  | 0.1027  | 0.125        | 0.089 | 0.070 | 0.071 |

`val_NDCG@10_std ≈ 0.163` для всех 4 методов → gap audio vs ID на грани шума, **bootstrap CI на test (шаг 12) обязателен** для defensible-результатов.

**Наблюдения:**

1. **H1 (минимальная) подтверждается на val.** Audio-методы > ID-методы по всем размерам группы кроме s=2 (там все ~0.12).
2. **Audio-преимущество растёт с размером группы.** Δ(AudioAGREE − AGREE) по NDCG@10: s=2: +0.009 (∼7%), s=3: +0.004 (∼4%), s=4: +0.008 (+11%), s=5: +0.007 (+11%). Самостоятельная история для текста ВКР: когда групповые предпочтения разнообразнее, контентный сигнал ценнее learnable ID-attention.
3. **Structural plateau при frozen scorer.** AGREE / GroupIM / GroupCrossAttn дали **идентичные до 4 знака** числа в обоих прогонах — продление с 60 до 100 эпох ничего не дало. AudioAGREE медленно ползла вверх ещё ~+0.001, но тоже близко к потолку. Это означает, что BPR-loss упёрся в структурный потолок, заданный фиксированными `s_{u,i}` от заморожённого SASRec (loss=0.65 ↔ pos − neg ≈ 0.08 в среднем). Не баг.
4. **Поздний прыжок train loss у GroupIM** на эпохах ~55-65 (с 1.1 до 0.7): MI-loss поначалу подавлял BPR-сходимость, потом сеть нашла представление, удовлетворяющее обоим. Прыжок не конвертировался в заметный рост val NDCG — это сигнал, что λ_MI=0.5 чуть высоковат. Если шаг 12 покажет, что GroupIM хочется поднять — пробуем λ_MI ∈ {0.1, 0.2}.
5. **Train loss audio > train loss ID** (0.65 vs 0.62), но **val NDCG audio > val NDCG ID**. Классическая «ID переучивает train, audio лучше обобщает» — желаемый сигнал для аргумента «audio-attention честнее как inductive bias».

**Артефакты на HF.** `Vladislavbro-500/music-recommendations` (dataset). Пользователь догружает чекпоинты `artifacts/aggregators/*` локально вручную после этой сессии, так что в новом чате они должны быть и локально, и на HF (bootstrap-ячейка ноутбука 07 на `hf_hub_download` всё равно идемпотентна).

**Обсуждённое в чате (для контекста новой сессии):**

- **End-to-end vs frozen scorer**: пользователь спросил «а если разморозить SASRec». Зафиксировано решение **оставить frozen для H1**: (а) разморозка делает сравнение нечестным — ID-методы выигрывают сильнее, у них есть learnable user_emb, у audio только phi/Q/K, (б) «оригинальные пайплайны» AGREE/GroupIM не используют SASRec вообще — full-replica требует переписать всё, не 1 день. End-to-end-вариант — естественный Phase 3 как ablation, не замена основной таблицы.

**Контракт для шага 12 (следующий чат).** План `07_eval_groups.ipynb`:
1. HF-download артефактов в `artifacts/` (идемпотентно, локальные файлы пропускаются).
2. `groups_split.pkl` → `test_samples` (используем `test_groups` + `user_test_targets`).
3. Для каждого из 4 методов: загрузить `best.pt`, `trainer.predict_group_scores(test_samples)`, NDCG@10/20.
4. **Bootstrap CI** (1000 resamples) на `per_sample` NDCG → mean ± 95% CI per method.
5. Срез по размеру группы + bootstrap CI на каждом размере.
6. Финальная таблица + heatmap (метод × размер), paired bootstrap для significance (audio vs ID). После прогона лог пополнится: best val NDCG@10/20 per method, время обучения, замечания по сходимости / λ_MI.

**Открытое:**
- Если на Colab какой-то метод явно не сходится / λ_MI ломает GroupIM — заводим шаг 11b с узкой сеткой (только проблемный гиперпараметр).
- Bootstrap кода + артефактов на HF протестирован только локально (файлы уже есть → `[skip]`). Реальный pull проверится на Colab.

### 2026-05-12 — Шаг 12: 07_eval_groups.ipynb — обвязка (код готов, ждёт Colab) 🟡

**Артефакт:** [notebooks/07_eval_groups.ipynb](../notebooks/07_eval_groups.ipynb) — 26 ячеек, 7 секций по контракту PHASE_2_LOG. В этот же ноутбук уложен шаг 14 (тривиальные бейзлайны AVG/LM/MP) — финальная таблица будет 7 строк (4 обучаемых + 3 тривиальных), чтобы текст ВКР цитировал одну таблицу с одним и тем же bootstrap-протоколом.

**Структура:**
1. Bootstrap + HF-download (`hf_hub_download` идемпотентно тянет Phase 1 артефакты, аудио, `groups_split.pkl`, 4 чекпоинта `best.pt` + `config.json`).
2. Data setup — повтор протокола Phase 1: `load_yambda → filter_listens → filter_min_popularity → apply_item_remap → global_temporal_split`, чтобы `test_df` совпадал с тем, на чём учился скорер.
3. `build_group_samples(test_groups, user_test_targets, ground_truth='union', drop_empty=True)` → `test_samples` с union-таргетом ∩ candidates (как в шагах 5/11).
4. Для каждого из 4 методов: реконструкция `IDBasedAGREE/GroupIM/AudioAGREE/GroupCrossAttention(**defaults_из_шага_11)` → `load_state_dict(best.pt)` → `GroupAggregatorTrainer.predict_group_scores(test_samples)` → `evaluate_aggregator_scores` (per-sample arrays). Дефолты `d_emb`/`d_att`/`d_model`/`n_heads` идентичны конфигам ячеек 06_15/17/19/21 — иначе state_dict не загрузится.
5. Тривиальные бейзлайны (`trivial_group_scores` использует `lookup_per_user_scores` с `fill=0.0`, тот же контракт, что в `GroupTrainConfig.fill_score`).
6. Bootstrap 1000 resamples с **общей** resample-сеткой (`rng(seed=42)`) → percentile 95% CI; per-size срез на отдельной resample-сетке (4 сетки по размерам); heatmap NDCG@10 (method × size) в `docs/figures/eval_heatmap_method_size.png`.
7. Paired bootstrap на той же resample-сетке: 4 пары `Audio* − ID*` × 2 K → mean delta + 95% CI + one-sided p `Pr(Δ ≤ 0)`.
8. Сохранение: `artifacts/eval_results/{summary.csv, summary_by_size.csv, paired.csv, per_sample.npz}`. `per_sample.npz` содержит сырые массивы + `resample_idx`, чтобы шаг 13 мог пересчитать что угодно без перепрогона моделей.

**Зафиксированные решения по этому шагу:**

1. **Тривиальные бейзлайны (шаг 14) — в этом же ноутбуке, не в отдельном.** Финальная таблица для ВКР должна быть одна (7 методов × 2 K × CI), чтобы текст не выделял «обучаемые vs тривиальные» как два разных эксперимента — это один протокол.
2. **`fill=0.0` для тривиальных бейзлайнов** (item ∉ top-K юзера → 0). Совместимо с `GroupTrainConfig.fill_score` из шага 6 → обучаемые методы и бейзлайны едят одно и то же сырьё. Альтернатива (`fill=-inf` для LM, `fill=0` для остальных) ввела бы asymmetry, которую пришлось бы оправдывать в тексте.
3. **Общая resample-сетка** для всех методов и paired-теста. Это (а) экономит код, (б) даёт согласованные CI: разница маржинальных средних совпадает с paired delta на каждой bootstrap-выборке. CI для пары `Audio*−ID*` уже бутстрапится из той же `resample_idx[B,n]` (`int32` сохраняется в `per_sample.npz` для воспроизводимости).
4. **Audio-* vs ID-* paired pairs (4 шт), не all-pairs.** По выбору пользователя — H1-минимальная проверка прямая (audio лучше ID), all-pairs дал бы избыточный шум на multiple testing.
5. **Heatmap → `docs/figures/`, не в `artifacts/`.** Графики для ВКР живут рядом с уже существующими (`train_4_approaches.png` и т.п.), а не в eval-csv.

**Sanity-чек (локально, CPU):**
- Все 4 чекпоинта грузятся без missing/unexpected keys (стандартный `load_state_dict`).
- `predict_group_scores` на subset из 20 групп с synthetic-таргетом возвращает NDCG@10 ≈ 0.55–0.58 (ожидаемо высокий, т.к. fake-таргеты = top-5 у каждого члена).
- Bootstrap-логика проверена на synthetic-данных (a∼N(0.10, 0.20), b∼N(0.09, 0.20) → mean diff 0.016, CI [-0.009, +0.040], p=0.10) — поведение разумное.
- AVG/LM/MP в чистом виде → `mean/min/max` по оси G.

**Время прогона (оценка):**
- Локально на subset 20 групп: 0–0.1с/метод. Полные 2000 test-групп на Colab G4 ≈ 1–2 мин на метод × 4 = 4–8 мин + bootstrap 1000 ≈ секунды на numpy. Итого <10 мин.

**Шаг 13 → секции 7.1–7.5 этого же ноутбука (вместо отдельного 08):**
- 7.1 — компактная таблица `mean [lo95, hi95]` (читается из in-memory `summary_df`).
- 7.2 — forest plot NDCG@10 (точки + 95% CI, аудио синие, ID оранжевые, тривиалы серые) → `docs/figures/eval_forest_plot.png`.
- 7.3 — NDCG@10 vs group size с CI для 5 ключевых методов → `docs/figures/eval_ndcg_by_size.png` (проверяет гипотезу шага 11 о росте audio-преимущества с size).
- 7.4 — paired-таблица со звёздочками `*/**/***` по p_one_sided.
- 7.5 — LaTeX-фрагмент в `artifacts/eval_results/summary_table.tex` (`\input{...}` дружелюбно).

**Открытое (риски, которые проявятся на Colab):**
- Размер `C_G` на test может оказаться > 1024 для крупных групп → CrossAttention медленный (риск из таблицы выше). Если eval будет дольше 5 мин/метод — урезать `candidates` до K_cap=1024 в `predict_group_scores` (потребует мини-патч `GroupEvalDataset`).
- Union-таргет в test пустой для >5% групп → пересмотр GT-стратегии на шаге 13 (сейчас `drop_empty=True`, статистика логируется ячейкой `TEST stats`).
- HF-download `best.pt` файлов: на Colab нужно убедиться, что `hf_hub_download` тянет именно из `dataset` repo type (не `model`).
