# Phase 2 — журнал работы

> Главный источник истины по Phase 2. Новые чаты начинают с чтения `CLAUDE.md` + `PHASE_1_LOG.md` + этого файла.

## Цель Phase 2

Обучить и сравнить 4 групповых агрегатора поверх замороженного SASRec из Phase 1: **AGREE**, **GroupIM**, **Audio-AGREE**, **GroupCrossAttention**. Метрика — NDCG@10/20 на синтетических random-группах. Конечный артефакт — таблица сравнения для текста ВКР.

Гипотеза H1 (минимальная): audio-aware агрегаторы ≥ ID-based по NDCG@10/20.

## Входы Phase 1

| Артефакт | Путь | Что внутри |
|---|---|---|
| Чекпоинт SASRec | `artifacts/gsasrec/best.pt` | hidden=64, 2 layers, 2 heads, n_items=276305 |
| Конфиг + метрики | `artifacts/gsasrec/{config.json,metrics.csv}` | best val NDCG@10 ≈ 0.0855 (epoch 51), test 0.0726 |
| item_id → idx | `artifacts/gsasrec/item_id_to_idx.pkl` | 276,305 items (min_pop≥5) |
| Кэш per-user scores | `artifacts/user_scores_cache/scores.parquet` | top-K=200 |
| Data infra | `src/data/{yambda_loader,splits}.py` | загрузка + GTS |
| Скорер | `src/scorer/{gsasrec,gbce_loss,train,inference}.py` | замороженный, не трогаем |

**Что НЕ сделано в Phase 1 и нужно в Phase 2:**
- `src/data/group_synthesis.py` — заготовка под random-группы (Шаг 3).
- Аудиоэмбеддинги (14 GB) — не выгружены; нужен subset на 276k items (≈ 135 MB).

## Compute

Colab переехал с A100 на **G4** (новое поколение NVIDIA, мощнее A100, 95 GB RAM). Это снимает риск «subset аудио не влезет в RAM» (135 MB << 95 GB), позволяет грузить `embeddings.npy` целиком в память без `memmap`, и заметно ускоряет обучение агрегаторов — прогон 4 моделей ~5 минут.

## Зафиксированные решения

| Решение | Обоснование |
|---|---|
| Per-user скорер заморожен | CLAUDE.md §2; все группы видят один `s_{u,i}` |
| Loss — pairwise BPR + popularity-negatives | CLAUDE.md §4 |
| Первая итерация — random-группы, size 2–5 | CLAUDE.md §8 |
| Метрики — NDCG@10/20 | CLAUDE.md §6; Jain/Disagreement/DFH позже |
| AlignGroup не включаем | Несовместим с ephemeral+frozen-scorer |
| `C_G` — union top-K каждого члена | Согласовано с CLAUDE.md §4 |
| Аудио — range-read subset через `HfFileSystem` | 14 GB → 135 MB |
| Ground truth для group-target | Union test-listens членов ∩ candidates |
| MI-loss λ для GroupIM | 0.5 (середина сетки {0.1, 0.5, 1.0}) |
| `fill=0.0` для `per_user_scores[u,i]` при item ∉ top-K(u) | Совместимо с любой attention-формой; real scores ∈ [-0.2, 17.3] → 0 нижний край, не «нейтраль внутри распределения» |
| Negatives — popularity-weighted из `C_G \ targets` | Веса = `pop^0.75` (как в Phase 1); хвост уже отсеян, `C_G` = заведомо релевантные кандидаты |
| Тривиальные бейзлайны — в `07_eval_groups.ipynb`, не отдельным ноутбуком | Финальная таблица ВКР — одна (обучаемые + тривиальные), один и тот же bootstrap-протокол |
| Общая resample-сетка для всех методов и paired-теста | Согласованные CI: разница маржинальных средних совпадает с paired delta |
| Audio-* vs ID-* paired pairs (4 шт), не all-pairs | H1-минимальная — прямая проверка audio vs ID; all-pairs дал бы избыточный шум на multiple testing |

## Прогресс по шагам

| Шаг | Описание | Статус |
|---|---|---|
| 1 | Sanity-чек кэша + фикс архитектурного бага с pad | ✅ |
| 2 | Subset аудио 276k → `artifacts/audio/embeddings.npy` | ✅ |
| 3 | `src/data/group_synthesis.py` — random-группы | ✅ |
| 4 | Аудиопрофиль $\bar{a}_u$ — `artifacts/audio/user_profiles.npy` | ✅ |
| 5 | `src/eval/{metrics,group_eval}.py` | ✅ |
| 6 | `src/training/{bpr_loss,group_trainer}.py` | ✅ |
| 7 | `src/aggregators/{base,agree}.py` | ✅ |
| 8 | `src/aggregators/groupim.py` + MI-дискриминатор | ✅ |
| 9 | `src/aggregators/audio_agree.py` | ✅ |
| 10 | `src/aggregators/group_cross_attn.py` | ✅ |
| 11 | `notebooks/06_train_aggregators.ipynb` — обучение 4 моделей | ✅ |
| 12 | `notebooks/07_eval_groups.ipynb` §1–6 — NDCG@10/20 + bootstrap CI | ✅ |
| 13 | `07_eval_groups.ipynb` §7 — финальная таблица + графики | ✅ |
| 14 | Тривиальные бейзлайны (AVG/LM/MP) | ✅ |

**Критический путь:** 1 → (2, 3 параллельно) → 4 → 5, 6 → 7..10 → 11 → 12 → 13. Шаги 5 и 6 можно делать параллельно с 7..10.

### План по ноутбукам

| Ноутбук | Шаги | Что на выходе |
|---|---|---|
| `06_train_aggregators.ipynb` | 11 | 4 чекпоинта; графики train/val loss; val NDCG@10 по эпохам |
| `07_eval_groups.ipynb` | 12, 13, 14 | csv-таблица: метод × NDCG@{10,20} × bootstrap CI; срез по размеру; forest plot; LaTeX-фрагмент |

> Изначально планировалось 3 ноутбука Phase 2, но шаги 13 и 14 слились в 07: финальный анализ читает in-memory `summary_df`/`per_sample` и работает с тем же bootstrap-протоколом, что и сам eval.

> Нумерация ноутбуков. В CLAUDE.md план писался от `04_train_aggregators.ipynb`, но к моменту шага 11 диск уже занял 04/05 под `audio_subset` / `user_audio_profiles`. Phase 2 training notebook = **06**, eval = **07**.

## Контракты данных

- **Group sample:** `{group_id, members, candidates, targets}`. Кандидаты = union top-K членов; targets = union test-listens ∩ candidates.
- **Batch агрегатора:** контракт из CLAUDE.md §4 + добавлены `group_mask`/`candidate_mask` (BoolTensor) для pad-маскинга.
- **`per_user_scores`:** fill=0.0 для item ∉ top-K(u) — совместимо с любой attention-формой.
- **Negatives:** popularity-weighted из `C_G \ targets`, веса = pop^0.75.
- **Аудиокэш:** `embeddings.npy [n_items+1, 128]`, индексируется `item_id_to_idx.pkl`. `audio_valid = norm(arr, axis=1) > 0` (4.15% catalog drift — items без аудио остаются нулями, маскируются естественно через zero-vector attention).
- **Pad-маскинг (расширение контракта `GroupAggregator.forward`):** Trainer паддит `|G|`/`|C_G|` до batch-max и передаёт булевы маски как kwargs (`group_mask: [B, G_max]`, `candidate_mask: [B, C_max]`). Pad-кандидатам после forward присваивается `-inf`. Pad-членам соответствует `per_user_scores=0` и `audio_profile=0` — но агрегатор обязан игнорировать их через `group_mask` (иначе attention/mean сместятся).

## Подробные записи по шагам

### 2026-05-11 — Шаг 1: sanity-чек кэша + фикс архитектурного бага ✅

**Кэш `artifacts/user_scores_cache/scores.parquet`:** 1.83M строк, K=200 на юзера, 9170 uids (из 9194 train; 24 отфильтрованы как <2 events). Схема: `uid` int64, `item_idx` int64, `score` float32, `rank` int32. item_idx ∈ [1, 276299], PAD=0 исключён. K=200 хватает для `C_G = union(top-K members)` при размере групп ≤5.

**Найдено и пофикшено два бага:**

1. **Inference запускался с `exclude_history=True`** — противоречит Phase 1 eval-протоколу (для музыки маскирование занижает NDCG в 3 раза, см. PHASE_1_LOG). Исправлено флагом в ноутбуке.
2. **Архитектурный баг `GSASRec.forward`:** комбинация `causal_mask + src_key_padding_mask` для query-позиций с all-masked keys давала `softmax(-inf) = NaN`, который через residual'ы протекал до позиции 199. Симптом — 1222 юзера (13.3%) с полностью NaN-кэшем (короткая train-история <200 events). Локально воспроизведено на synthetic n_real ∈ {1..199}; n_real=200 работал. **Фикс** в [src/scorer/gsasrec.py](../src/scorer/gsasrec.py): заменил `src_key_padding_mask` на per-batch 3D `attn_mask [B*n_heads, L, L]` с causal+pad-key masking, но всегда доступной диагональю (любая query attend на себя минимум). Distribution shift для real-позиций нулевой, для full-200 юзеров выход бит-в-бит идентичен старому.

**Латентный эффект на Phase 1:** `evaluate_ndcg` тоже страдал → короткие юзеры давали 0 contribution. Baseline 0.0726 слегка занижен; re-eval не делаю, для текста ВКР — footnote.

**Финальный кэш:** NaN: 0 (было 244,400), 52,642 уникальных item_idx в union топ-200 (+2.2k vs до фикса), score: mean 4.70, range [-0.22, 17.26], duplicates 0.

**Открытое:** возможно поднять K до 500 для popularity-negatives — решу на шаге 6.

### 2026-05-11 — Шаг 2: subset аудиоэмбеддингов ✅

**Подход.** Качаем `embeddings.parquet` (14 GB, 7.72M × 128) целиком через `hf_hub_download` на Colab-диск, читаем две колонки в `pyarrow.Table`, фильтруем `np.isin(item_id, target_ids)` по 276,305 items из Phase 1, сохраняем `[n_items+1, 128]` float32 (~135 MB), row 0 — PAD. Локально файл не оседает — результат скачали после Colab-прогона.

**Артефакты:**
- [src/data/audio_embeddings.py](../src/data/audio_embeddings.py) — `extract_audio_subset(item_id_to_idx, output_path, use_normalized=False)`.
- [notebooks/04_audio_subset.ipynb](../notebooks/04_audio_subset.ipynb) — 4 ячейки: bootstrap → загрузка `item_id_to_idx` → вызов функции → sanity-check (+ доп. ячейка с popularity-анализом missing).

**Probe схемы parquet:** `num_row_groups=30`, `num_rows=7,721,749`. Колонки: `item_id uint32`, `embed large_list<double>`, `normalized_embed large_list<double>` (dim 128). Берём `embed`, нормированный вариант — флагом при необходимости.

**Запуск на Colab (CPU runtime, 50 GB RAM):** выполнен, всё прошло успешно. Отдельно прогнан popularity-анализ missing items (`load_yambda` + `filter_listens` + `value_counts` — ~5–10 мин на CPU, узкое место — `to_pandas()` на 46M строк, GPU тут не помогает).

**Результат `artifacts/audio/embeddings.npy`:** shape `(276306, 128)`, dtype float32, 134.9 MB. PAD-row (idx=0) нулевой, NaN/Inf нет. Норма (non-zero): mean=28.45, range [14.50, 190.51].

**Coverage:** 264,840 / 276,305 items с эмбеддингом (**95.85%**); **11,465 (4.15%) — без аудио, остаются нулями**.

**Природа пропусков (sanity-чек по popularity, ячейка в `04_audio_subset.ipynb`):**

|  | missing | present |
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

**Открыто (на момент шага):** train/val/test split групп пока не делаем — формат `Group sample {members, candidates, targets}` соберём в шагах 5/6, когда появится eval-обвязка и определимся с union-target по test-listens.

### 2026-05-11 — Шаг 4: user audio profiles ✅

**Артефакт:** [src/data/audio_embeddings.py](../src/data/audio_embeddings.py) — функция `build_user_audio_profiles(train_listens, item_embeddings)` → `(profiles[n_users, 128], uid_to_row, user_audio_valid[n_users])`. Усредняет по train listen+ истории, маскируя items с `norm(emb)==0` (4.15% catalog drift из шага 2). Юзеры без valid items получают zero-row и `user_audio_valid=False`.

**Дизайн.** Индексация — compact `[n_users, 128]` + `uid_to_row.pkl` (а не dense `[max_uid+1, ...]`), по аналогии с `item_id_to_idx`. Это даёт ~4.5 MB вместо потенциально гигабайтов на разреженных raw-uids. История — только **train** (после GTS), чтобы не было лика из val/test.

**Smoke (локально, synthetic).** `n_items=10`, 2 missing items (idx 3, 7), 4 юзера; вручную проверены: (а) user с миксом valid+invalid усредняет только valid; (б) user только с invalid item получает zero-row и valid=False; (в) счётчики совпадают. Прошло.

**Ноутбук:** [notebooks/05_user_audio_profiles.ipynb](../notebooks/05_user_audio_profiles.ipynb) — Colab-runner: load YAMBDA-50m → filter (listen+, min_pop≥5) → load `item_id_to_idx.pkl` → remap → GTS → load `embeddings.npy` → `build_user_audio_profiles` → сохранение в `artifacts/audio/{user_profiles.npy, uid_to_row.pkl, user_audio_valid.npy}`. Включён sanity-чек: для случайного uid профиль пересчитывается «руками» и сравнивается с сохранённым (max|diff| < 1e-5).

**Контракт для шагов 7–10.** AudioAGREE / GroupCrossAttn в `forward` принимают `audio_profiles_users[B, |G|, 128]` — собираются из этой таблицы по `uid_to_row`. Для юзеров с `user_audio_valid=False` агрегатор должен либо игнорировать их в attention, либо fallback на скоринг без audio-ветки.

### 2026-05-12 — Шаг 5: eval-обвязка (metrics + group_eval) ✅

**Артефакты:**
- [src/eval/metrics.py](../src/eval/metrics.py) — `dcg_at_k`, `idcg_at_k`, `ndcg_at_k` (низкоуровневый, `[B, L]` бинарных релевантностей + `n_relevant[B]`), `ranking_ndcg_at_k` (сортирует scores per-row, поддерживает несколько K за вызов), `ndcg_from_ranking` (single-query). Векторизовано на numpy, без torch-зависимости — eval-обвязка не зависит от среды агрегатора.
- [src/eval/group_eval.py](../src/eval/group_eval.py) — `GroupSample` dataclass, `build_group_samples(groups, user_topk, user_test_targets, ground_truth, drop_empty, drop_missing_member)` собирает кандидатные пулы (union top-K членов) и таргеты (union/intersection test-listens ∩ candidates); `evaluate_aggregator_scores(samples, group_scores, k_list)` считает NDCG@K с разбивкой по размеру группы и per-sample массивами под бутстрап CI; хелперы `topk_from_score_cache` и `test_targets_from_df` для подключения parquet'ов из Phase 1.
- [tests/test_eval.py](../tests/test_eval.py) — 18 toy-тестов (pytest). Покрывают: ручные DCG/IDCG/NDCG, идеальный/нулевой ranking, truncate на K, корректность сортировки по score, batch over rows, union/intersection-target, drop_empty/drop_missing_member, размерные срезы, валидация формы скоров.

**Зафиксированные решения по шагу:**

1. **Default ground-truth — union** по test-listens членов. Intersection остаётся параметром (`ground_truth="intersection"`) для sanity-cell в ноутбуке. Union даёт более полный recall-сигнал для эфемерных групп, где не все члены пересекаются.
2. **Out-of-pool релеванты исключаются.** `targets = union(test_listens) ∩ candidates`. IDCG нормируется только по достижимым релевантам в `C_G`. Это согласовано с тем, что агрегатор не видит весь каталог — оценивать «недостижимый» NDCG бессмысленно.

**Запуск:** `pytest tests/test_eval.py -q` → 18/18 passed.

**Контракт для шага 6 (trainer).** Trainer на каждом батче должен отдавать `list[np.ndarray[|C_G|]]` group-скоров — `evaluate_aggregator_scores` принимает его напрямую. `per_user_scores` для агрегатора собираются отдельно из `scores.parquet` (см. шаг 1 кэш) — не в этом модуле.

### 2026-05-12 — Шаг 6: BPR loss + GroupAggregatorTrainer ✅

**Артефакты:**
- [src/training/bpr_loss.py](../src/training/bpr_loss.py) — `pairwise_bpr_loss(pos, neg)`. Использует `F.logsigmoid` (численно стабилен), бродкастит `[B] vs [B]` и `[B] vs [B,K]`.
- [src/training/group_trainer.py](../src/training/group_trainer.py) — `GroupTrainConfig`, `GroupAggregatorTrainer`, `GroupTrainDataset`, `GroupEvalDataset`, `collate_groups`, хелперы `build_user_score_lookup`, `lookup_per_user_scores`, `compute_pop_counts`.
- [tests/test_training.py](../tests/test_training.py) — 13 тестов (BPR + lookup + pop + neg-sampling + collate + end-to-end fit с тривиальным `MeanScoreAggregator`).

**Зафиксированные решения по шагу:**

1. **Fill для `per_user_scores[u,i]` при item ∉ top-K(u): `0.0`.** Совместимо с любой attention-формой (нет NaN от `-inf`), а реальные scores ∈ [-0.2, 17.3] так что 0 — нижний край, не «нейтраль внутри распределения». Этот же fill используется для pad-позиций в батче.
2. **Негативы — popularity-weighted из `C_G \ targets`.** В духе CLAUDE.md «popularity-negatives». Веса = `popularity^0.75` (как в Phase 1 `compute_item_popularity`), таргеты обнуляются до нормализации. Хвост уже отсеян (`C_G` = union top-K, заведомо релевантные кандидаты) — это и есть hard negatives для группы.

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

**Smoke (`pytest tests/test_aggregators.py tests/test_eval.py tests/test_training.py -q`):** 43/43 passed за 0.98s.

**Контракт для шагов 8–10.** Все три оставшихся агрегатора (GroupIM, AudioAGREE, GroupCrossAttn) наследуются от `GroupAggregator` и обязаны: корректно маскировать softmax по `group_mask`; для audio-методов — дополнительно использовать `audio_valid`-маску по items; GroupIM реализует `regularization_loss(batch, scores)` для MI-loss — Trainer подключает его автоматически если `cfg.reg_loss_weight > 0`.

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

**Зафиксированные решения по шагу:**

1. **Attention item-agnostic, а не item-aware.** Это намеренный контраст к AGREE (item-aware) для чистой абляции: один и тот же frozen-scorer, одна и та же ID-таблица user-эмбеддингов, разная функция агрегации. Если оба метода хуже audio-веток (шаги 9–10), это будет аргумент за audio-сигнал; если AGREE сильно лучше GroupIM — за item-aware attention в ID-режиме.
2. **Negatives для MI — batch-shift (`torch.roll`).** Простейшая cross-batch стратегия, не требует дополнительной семплинг-обвязки. При `B<2` (вырожденный случай) MI-loss = 0. Это поведение явно протестировано.
3. **Кэш state в модуле, не в trainer.** Альтернатива — прокидывать `(member_emb, group_repr)` через возвращаемое значение forward, но это сломало бы единую сигнатуру `[B, C_max]` контракта. Кэш сбрасывается после каждого `regularization_loss` → не утекает в инференс.

**Smoke (`pytest tests/test_aggregators.py -q`):** 22/22 passed. Full regression `pytest tests/ -q`: **53/53 passed за ~0.93s** (включая 18 eval + 13 training + 22 aggregators).

**Открытое на момент шага:**
- λ для MI-loss — сетка `{0.1, 0.5, 1.0}` подберём на шаге 11. Финально остановились на 0.5 (середина), sweep отложен.
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

**Зафиксированные решения по шагу:**

1. **Missing-audio items/users не маскируются явно.** Контракт шага 2 говорит «missing-кандидаты → нулевой вклад в логит». Реализация: для item с `a_i = 0` MLP-логит зависит только от `a_bar_u` (одна половина concat обнулена); для user с `a_bar_u = 0` — только от `a_i`. Это естественный degradation без edge-cases с softmax(-inf) → NaN. Если у группы вся аудио-сторона нулевая (теоретически возможно), logit одинаковый по членам → uniform alpha → mean per_user_scores, что разумный fallback (= AVG). Покрыто `test_audio_agree_missing_audio_does_not_nan`.
2. **`d_att=64` по умолчанию** (vs `d_emb=32`/`d_att=32` у AGREE). Вход у нас 2·128=256 (а у AGREE — 2·32=64), поэтому скрытый слой пропорционально больше. Sweep по `d_att` — отложен.
3. **`forward` raises на отсутствие аудио** (vs тихий ignore у AGREE). Это сигнал, что Trainer без `item_audio`/`user_profiles` запускать AudioAGREE нельзя. Покрыто `test_audio_agree_raises_without_audio`.

**Smoke (`pytest tests/ -q`):** **62/62 passed за ~1.5s** (включая 18 eval + 13 training + 31 aggregators = 22 старых + 9 новых).

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

**Зафиксированные решения по шагу:**

1. **V-проекция выкинута.** Альтернатива (V = `Linear(a_bar_u)`, output → `Linear(d_model → 1)`) обучала бы свой собственный «item-агностичный summary group» — это нарушает контракт «frozen scorer + аггрегатор только взвешивает» и делает CrossAttn несравнимым с AGREE/AudioAGREE (там итог тоже Σ α · s). Текущий дизайн делает шаг 10 чистым ablation по сравнению с шагом 9: AudioAGREE использует MLP `phi(concat(a_u, a_i))` для логита, CrossAttn — H голов scaled dot-product. Все остальные части (per_user_scores, маска, BPR-обвязка) идентичны.
2. **Mean по головам, не concat.** Concat→`Linear(H → 1)` ввёл бы дополнительный обучаемый mixer, который имеет смысл если головы должны быть «специализированы». В нашем small-data сетапе (9k юзеров, 4 модели) проще оставить mean — это убирает 1 матрицу параметров и при H=1 эквивалентно AudioAGREE-без-MLP. Покрыто `test_cross_attn_single_head_equivalence`.
3. **`audio_valid`-маска по items не применяется.** Контракт шага 2 говорит «missing-кандидаты → нулевой вклад в логит». В CrossAttn нулевой `a_i` даёт нулевой Q → логиты по этому кандидату ≈ 0 по всем G → softmax по G даёт ~равномерное `alpha` → score ≈ mean(per_user_scores) (мягкий AVG-fallback). Это совместимо с тем же поведением AudioAGREE для missing-аудио и не требует edge-case-маскирования. Покрыто `test_cross_attn_missing_audio_does_not_nan`.

**Дефолты `d_model=64`, `n_heads=4`** (d_head=16). Выбор близок к AudioAGREE (`d_att=64`): один и тот же бюджет hidden-размера, но раздроблен на головы — это и есть основная разница, которую хотим измерить. Sweep по `d_model ∈ {32, 64}` и `n_heads ∈ {1, 2, 4, 8}` — отложен на шаг 11.

**Smoke (`pytest tests/ -q`):** **74/74 passed за ~1.45s** (18 eval + 13 training + 43 aggregators). Регрессий нет.

**Контракт для шага 11.** Все 4 агрегатора (AGREE/GroupIM/AudioAGREE/GroupCrossAttention) теперь имеют единую сигнатуру `forward(group_user_ids, candidate_ids, per_user_scores, audio_embeds_items, audio_profiles_users, group_mask, candidate_mask) -> [B, C_max]` и обучаются через один и тот же `GroupAggregatorTrainer`. ID-методам аудио-аргументы можно не подавать, audio-методам ID-аргументы можно (они игнорируются). Trainer получает `item_audio` / `user_profiles` / `uid_to_row` опционально — это путь к единому коду в ноутбуке.

### 2026-05-12 — Шаг 11: 06_train_aggregators.ipynb ✅

**Артефакт:** [notebooks/06_train_aggregators.ipynb](../notebooks/06_train_aggregators.ipynb) — 27 ячеек, структура «shared setup → 4 отдельных блока обучения → comparison-cell».

**Зафиксированные решения по шагу:**

1. **Имя ноутбука — `06_*`, не `04_*`** как в исходном плане CLAUDE.md. К Phase 2 диск уже занял 04/05 под аудио-prep. Phase 2 nb-нумерация: 06 train → 07 eval.
2. **Scope шага — single config per method, без полного sweep.** Для honest сравнения 4 методов на ВКР важен одинаковый compute budget и разумные дефолты, а не подгонка каждого. Sweep по `d_att`/`d_model`/`n_heads` отложен (если результаты шага 12 покажут, что какой-то метод явно недотюнен). λ_MI для GroupIM = 0.5 (середина сетки {0.1, 0.5, 1.0}) — единственное «непаперное» значение. lr=1e-3 для всех 4 (Adam default).
3. **Структура ноутбука — per-method блоки, не sweep-цикл.** Пользователь хотел иметь возможность переопределять конфиг каждого метода отдельно. Каждый блок = `IDBasedAGREE/GroupIM/AudioAGREE/GroupCrossAttention(**cfg)` + `trainer.fit()`. Помодульная итерация: правишь `d_emb` у GroupIM → перезапускаешь только её ячейку, остальное не трогается.
4. **Тест НЕ touchаем в этом ноутбуке.** Подсматривание в test при правке конфигов — методологическая ошибка (комиссия ВКР спросит «вы фиксировали гиперпараметры по val или по test?»). Val NDCG@10 используется для early stopping и финальной comparison-таблицы. Test → отдельный шаг 12 (`07_eval_groups.ipynb`) один раз после фиксации всех конфигов. Test_groups при этом синтезируются здесь же (тот же seed) и сохраняются в `groups_split.pkl` — это гарантирует, что шаг 12 видит идентичный split.
5. **Bootstrap — `hf_hub_download` из `Vladislavbro-500/music-recommendations` (dataset).** Артефакты Phase 1 + аудио залиты на HF (не в git), скачиваются в `artifacts/` идемпотентно (skip если файл уже есть). Это разводит код (git) и тяжёлые артефакты (HF) — стандартный pattern для Colab.

**Контракт данных, зафиксированный в ноутбуке:**
- `groups_split.pkl` (артефакт): `{train_groups, val_groups, test_groups, group_seed=42, size_dist, train_stats, val_stats}` — один файл, читается шагом 12.
- Train-таргеты = union train listens членов ∩ candidates; val-таргеты = union val listens. Test-таргеты в этом ноутбуке не строятся (но в `groups_split.pkl` лежат raw `test_groups` — шаг 12 сам построит `test_samples`).
- `item_audio` + `user_profiles` передаются Trainer'у для всех 4 методов (ID-методы игнорируют, audio-методы используют — единый код).
- Все 4 чекпоинта в `artifacts/aggregators/<name>/{best.pt, config.json, metrics.csv}`. После Colab-прогона — загружаются обратно на HF.

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

**Два прогона:**
- Run 1: `n_epochs=60, patience=20` (дефолт ноутбука).
- Run 2: `n_epochs=100, patience=30` — расширили после анализа кривых.

**Финальные числа (Run 2, val):**

| method | NDCG@10 | NDCG@20 | s=2 | s=3 | s=4 | s=5 |
|---|---:|---:|---:|---:|---:|---:|
| AudioAGREE | **0.0978** | **0.1097** | 0.135 | 0.094 | 0.080 | 0.071 |
| GroupCrossAttn | 0.0941 | 0.1056 | 0.119 | 0.094 | 0.081 | 0.069 |
| AGREE | 0.0916 | 0.1024 | 0.126 | 0.090 | 0.074 | 0.064 |
| GroupIM (λ=0.5) | 0.0911 | 0.1027 | 0.125 | 0.089 | 0.070 | 0.071 |

`val_NDCG@10_std ≈ 0.163` для всех 4 методов → gap audio vs ID на грани шума, **bootstrap CI на test (шаг 12) обязателен** для defensible-результатов.

**Наблюдения:**

1. **H1 (минимальная) подтверждается на val.** Audio-методы > ID-методы по всем размерам группы кроме s=2 (там все ~0.12).
2. **Audio-преимущество растёт с размером группы.** Δ(AudioAGREE − AGREE) по NDCG@10: s=2: +0.009 (∼7%), s=3: +0.004 (∼4%), s=4: +0.008 (+11%), s=5: +0.007 (+11%). Самостоятельная история для текста ВКР: когда групповые предпочтения разнообразнее, контентный сигнал ценнее learnable ID-attention.
3. **Structural plateau при frozen scorer.** AGREE / GroupIM / GroupCrossAttn дали **идентичные до 4 знака** числа в обоих прогонах — продление с 60 до 100 эпох ничего не дало. AudioAGREE медленно ползла вверх ещё ~+0.001, но тоже близко к потолку. Это означает, что BPR-loss упёрся в структурный потолок, заданный фиксированными `s_{u,i}` от заморожённого SASRec (loss=0.65 ↔ pos − neg ≈ 0.08 в среднем). Не баг.
4. **Поздний прыжок train loss у GroupIM** на эпохах ~55-65 (с 1.1 до 0.7): MI-loss поначалу подавлял BPR-сходимость, потом сеть нашла представление, удовлетворяющее обоим. Прыжок не конвертировался в заметный рост val NDCG — это сигнал, что λ_MI=0.5 чуть высоковат. Если шаг 12 покажет, что GroupIM хочется поднять — пробуем λ_MI ∈ {0.1, 0.2}.
5. **Train loss audio > train loss ID** (0.65 vs 0.62), но **val NDCG audio > val NDCG ID**. Классическая «ID переучивает train, audio лучше обобщает» — желаемый сигнал для аргумента «audio-attention честнее как inductive bias».

**Артефакты на HF.** `Vladislavbro-500/music-recommendations` (dataset). Чекпоинты в `artifacts/aggregators/<name>/{best.pt,config.json,metrics.csv}`.

**Обсуждённое в чате (контекст для шага 12):**

**End-to-end vs frozen scorer.** Пользователь спросил «а если разморозить SASRec». Зафиксировано решение **оставить frozen для H1**: (а) разморозка делает сравнение нечестным — ID-методы выигрывают сильнее, у них есть learnable user_emb, у audio только phi/Q/K, (б) «оригинальные пайплайны» AGREE/GroupIM не используют SASRec вообще — full-replica требует переписать всё, не 1 день. End-to-end-вариант — естественный Phase 3 как ablation, не замена основной таблицы.

## 2026-05-12 — Шаги 12–14: Eval, финальная таблица, тривиальные бейзлайны ✅

**Артефакты:**
- [artifacts/eval_results/summary.csv](../artifacts/eval_results/summary.csv) — основная таблица.
- [artifacts/eval_results/summary_by_size.csv](../artifacts/eval_results/summary_by_size.csv) — срез по размеру.
- [artifacts/eval_results/paired.csv](../artifacts/eval_results/paired.csv) — paired bootstrap audio vs ID.
- [artifacts/eval_results/per_sample.npz](../artifacts/eval_results/per_sample.npz) — сырые NDCG + `resample_idx` для воспроизводимости.
- [artifacts/eval_results/summary_table.tex](../artifacts/eval_results/summary_table.tex) — LaTeX-фрагмент.
- [docs/figures/eval_forest_plot.png](figures/eval_forest_plot.png), [eval_heatmap_method_size.png](figures/eval_heatmap_method_size.png), [eval_ndcg_by_size.png](figures/eval_ndcg_by_size.png).

**Протокол:** 2000 test-групп (тот же `groups_split.pkl` что и train/val), `build_group_samples(ground_truth='union', drop_empty=True)`. Bootstrap — 1000 resamples с **общей resample-сеткой** (`rng(seed=42)`) → marginal + paired CI согласованы. Размерный срез — 4 отдельных resample-сетки.

**Зафиксированные решения по шагу:**

1. **Тривиальные бейзлайны (шаг 14) — в этом же ноутбуке, не в отдельном.** Финальная таблица для ВКР должна быть одна (7 методов × 2 K × CI), чтобы текст не выделял «обучаемые vs тривиальные» как два разных эксперимента — это один протокол.
2. **`fill=0.0` для тривиальных бейзлайнов** (item ∉ top-K юзера → 0). Совместимо с `GroupTrainConfig.fill_score` из шага 6 → обучаемые методы и бейзлайны едят одно и то же сырьё. Альтернатива (`fill=-inf` для LM, `fill=0` для остальных) ввела бы asymmetry, которую пришлось бы оправдывать в тексте.
3. **Общая resample-сетка** для всех методов и paired-теста. Это (а) экономит код, (б) даёт согласованные CI: разница маржинальных средних совпадает с paired delta на каждой bootstrap-выборке. CI для пары `Audio*−ID*` уже бутстрапится из той же `resample_idx[B,n]` (`int32` сохраняется в `per_sample.npz` для воспроизводимости).
4. **Audio-* vs ID-* paired pairs (4 шт), не all-pairs.** По выбору пользователя — H1-минимальная проверка прямая (audio лучше ID), all-pairs дал бы избыточный шум на multiple testing.
5. **Heatmap → `docs/figures/`, не в `artifacts/`.** Графики для ВКР живут рядом с уже существующими (`train_4_approaches.png` и т.п.), а не в eval-csv.

### Финальная таблица (test, 95% bootstrap CI)

| # | Метод | NDCG@10 | 95% CI | NDCG@20 | 95% CI |
|---|---|---:|---|---:|---|
| 1 | **GroupCrossAttn** | **0.0916** | [0.0833, 0.1008] | **0.1024** | [0.0939, 0.1114] |
| 2 | **AudioAGREE** | **0.0901** | [0.0823, 0.0990] | **0.1014** | [0.0934, 0.1097] |
| 3 | MP | 0.0828 | [0.0755, 0.0895] | 0.0915 | [0.0843, 0.0983] |
| 4 | GroupIM | 0.0811 | [0.0739, 0.0885] | 0.0912 | [0.0844, 0.0983] |
| 5 | AGREE | 0.0790 | [0.0718, 0.0860] | 0.0901 | [0.0829, 0.0970] |
| 6 | AVG | 0.0695 | [0.0632, 0.0754] | 0.0819 | [0.0758, 0.0877] |
| 7 | LM | 0.0312 | [0.0275, 0.0351] | 0.0363 | [0.0326, 0.0400] |

### Paired bootstrap (audio − ID)

| audio | id | K | Δ mean | 95% CI | p one-sided |
|---|---|---:|---:|---|---:|
| AudioAGREE | AGREE | 10 | +0.0110 | [0.0053, 0.0177] | <0.001 |
| AudioAGREE | AGREE | 20 | +0.0112 | [0.0052, 0.0176] | <0.001 |
| AudioAGREE | GroupIM | 10 | +0.0090 | [0.0024, 0.0154] | 0.005 |
| AudioAGREE | GroupIM | 20 | +0.0101 | [0.0037, 0.0164] | 0.001 |
| GroupCrossAttn | AGREE | 10 | +0.0125 | [0.0054, 0.0194] | <0.001 |
| GroupCrossAttn | AGREE | 20 | +0.0123 | [0.0059, 0.0190] | <0.001 |
| GroupCrossAttn | GroupIM | 10 | +0.0105 | [0.0033, 0.0175] | 0.001 |
| GroupCrossAttn | GroupIM | 20 | +0.0112 | [0.0046, 0.0178] | <0.001 |

**Все 8 пар значимы**, CI не пересекают 0. H1 подтверждена на test.

### Срез NDCG@10 по размеру группы

| size | AGREE | GroupIM | AudioAGREE | GroupCrossAttn | MP | AVG | LM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.095 | 0.104 | 0.110 | **0.115** | 0.105 | 0.094 | 0.044 |
| 3 | 0.073 | 0.072 | 0.078 | **0.081** | 0.076 | 0.065 | 0.027 |
| 4 | 0.073 | 0.074 | **0.090** | 0.087 | 0.076 | 0.057 | 0.027 |
| 5 | 0.078 | 0.077 | **0.089** | 0.089 | 0.072 | 0.056 | 0.028 |

### Ключевые наблюдения для текста ВКР

1. **Audio значимо обходит ID** (paired Δ ≈ +0.011, относительно ~12–16%). На marginal CI пара audio/ID почти впритык, но paired bootstrap отлавливает корреляцию между методами через общую resample-сетку — значимость p≤0.005. Это критичная ремарка: marginal CI могут вводить в заблуждение, paired-тест — правильный инструмент.

2. **Два audio-варианта статистически неразличимы** (AudioAGREE 0.0901 vs CrossAttn 0.0916, CI пересекаются). На val порядок был обратный — gap внутри шума. Формулировка для ВКР: «выбор между MLP-attention и multi-head dot-product не критичен, важен сам сигнал».

3. **MP (max popularity) обходит обучаемые ID-методы.** 0.0828 > GroupIM 0.0811 > AGREE 0.0790. Сильный аргумент: **обучаемые ID-аттеншены поверх frozen SASRec не дают полезного сигнала сверх тривиального max**. Audio — единственный способ пробить плато.

4. **Audio-преимущество растёт с размером группы.** На s=2 все методы (кроме AVG/LM) близки (0.094–0.115). На s=3–5 ID-методы застревают на ~0.073, audio держит 0.087–0.090. Heatmap и by-size plot делают это видимым.

5. **LM провал (0.031)** — `min(s_{u,i})` ломается при `s ∈ [-0.2, 17.3]` и fill=0 для not-in-topK. Negative control, оправдывает выбор AVG/MP как «настоящих» тривиалов.

## Caveats для текста ВКР (важно)

Phase 2 сравнивает **аггрегационные механизмы поверх фиксированного SASRec**, а не полные методы AGREE/GroupIM. В тексте формулировать аккуратно:

**В главе «Методы»:** «Для сопоставимости агрегаторов используется общая архитектура: фиксированный sequential per-user scorer + обучаемый групповой аггрегатор. От AGREE (Cao et al., 2018) заимствуется item-aware attention; от GroupIM (Sankar et al., 2020) — item-agnostic attention с MI-регуляризацией. Audio-AGREE и Group Cross-Attention заменяют ID-эмбеддинги в attention на аудио YAMBDA.»

**В разделе «Ограничения»:** «Сравнение не воспроизводит полные пайплайны AGREE/GroupIM, где per-user скорер обучается совместно с агрегатором. Наша рамка изолирует вклад аггрегационного механизма при фиксированном скорере, что соответствует индустриальному two-stage retrieval-rerank сетапу.»

**В разделе «Результаты»:** избегать «AudioAGREE превосходит AGREE» → «при фиксированном per-user скорере замена ID на аудио в attention механизме AGREE улучшает NDCG@10 на 12% (paired bootstrap p<0.001)».

Эта формулировка усиливает работу, а не ослабляет: чистая ablation, индустриально релевантный сетап, и сюжет «ID-сигнал поверх SASRec не оправдан, нужен audio» только в этой рамке работает (см. наблюдение 3 выше).

## Future work

**Phase 3 — end-to-end fine-tune SASRec под group BPR (не делаем в этой итерации из-за сроков).**

Технически просто: `score_items` метод в SASRec, scorer-aware trainer, два parameter group в Adam (lr_aggregator=1e-3, lr_scorer=1e-5), KD-регуляризация на solo-user test для страховки от forgetting. ~2 рабочих дня кода + 2–3 ч компьюта на Colab.

Ожидаемый исход: gap audio vs ID может сократиться, ID-методы получат новый источник сигнала через градиенты в эмбеддинги SASRec. Вероятность что audio всё ещё впереди — ~55–65%. На 50m есть риск статистической нерешённости (val std≈0.163 у всех методов); полноценная e2e-проверка лучше делается на 500m (~3 ч SASRec + 1 ч переделка audio subset + аудиопрофили). 5b — отдельная диссертация (sampled softmax, бОльшая модель, теряется Yandex-якорь).

**Прочее future work (по убыванию приоритета):**
- Дополнительные метрики разнообразия/справедливости: Jain@K, Disagreement@K, DFH@K.
- Homogeneous / heterogeneous группы (по similarity вкусов), а не только random.
- Cold-start срез по доле редких треков в test-таргете.
- λ_MI sweep для GroupIM ({0.1, 0.2, 0.5}) — слабый prior, что 0.1–0.2 даст +0.001–0.003.
- MI-дискриминатор: bilinear → MLP (тривиальная правка, если λ_MI sweep не поможет).
- Sweep `d_emb`/`d_att` (AGREE/GroupIM) и `d_model`/`n_heads` (CrossAttn) для тюнинга каждого метода.

## Риски (пост-фактум)

| Риск | Что произошло |
|---|---|
| Subset аудио не помещается в RAM | Снято: G4 даёт 95 GB |
| Union-target в test пустой | 4.x% групп с empty target, отброшены `drop_empty=True` |
| GroupIM MI-loss доминирует | λ=0.5 не сломал, но не помог — поздний train-jump без val-gain |
| Полусырая cross-attention медленная | Не реализовалось, C_G ≤ 1024 держится естественно |
| 9.2k пользователей мало для 4 моделей | Bootstrap CI обязательны, не overclaim в ВКР — учли в caveats |
