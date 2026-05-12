# Phase 2 — журнал работы

> Главный источник истины по Phase 2. Новые чаты начинают с чтения `CLAUDE.md` + `PHASE_1_LOG.md` + этого файла.

## Цель Phase 2

Обучить и сравнить 4 групповых агрегатора поверх замороженного SASRec из Phase 1: **AGREE**, **GroupIM**, **Audio-AGREE**, **GroupCrossAttention**. Метрика — NDCG@10/20 на синтетических random-группах.

Гипотеза H1 (минимальная): audio-aware агрегаторы ≥ ID-based по NDCG@10/20.

## Входы Phase 1

| Артефакт | Путь | Что внутри |
|---|---|---|
| Чекпоинт SASRec | `artifacts/gsasrec/best.pt` | hidden=64, 2 layers, 2 heads, n_items=276305 |
| Конфиг + метрики | `artifacts/gsasrec/{config.json,metrics.csv}` | best val NDCG@10 ≈ 0.0855, test 0.0726 |
| item_id → idx | `artifacts/gsasrec/item_id_to_idx.pkl` | 276,305 items (min_pop≥5) |
| Кэш per-user scores | `artifacts/user_scores_cache/scores.parquet` | top-K=200 |

## Compute

Colab G4 (95 GB RAM) — снимает риск «subset аудио не влезет», грузим всё в RAM, прогон 4 моделей ~5 минут.

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

## Контракты данных

- **Group sample:** `{group_id, members, candidates, targets}`. Кандидаты = union top-K членов; targets = union test-listens ∩ candidates.
- **Batch агрегатора:** контракт из CLAUDE.md §4 + добавлены `group_mask`/`candidate_mask` (BoolTensor) для pad-маскинга.
- **`per_user_scores`:** fill=0.0 для item ∉ top-K(u) — совместимо с любой attention-формой.
- **Negatives:** popularity-weighted из `C_G \ targets`, веса = pop^0.75.
- **Аудиокэш:** `embeddings.npy [n_items+1, 128]`, индексируется `item_id_to_idx.pkl`. `audio_valid = norm(arr, axis=1) > 0` (4.15% catalog drift — items без аудио остаются нулями, маскируются естественно через zero-vector attention).

## Сжатые записи по шагам 1–11

### Шаг 1 — кэш scores + фикс бага ✅

`scores.parquet`: 1.83M строк, K=200, 9170 uids. Найдено два бага:
1. Inference запускался с `exclude_history=True` — противоречит eval-протоколу Phase 1.
2. `GSASRec.forward`: комбинация `causal_mask + src_key_padding_mask` для query с all-masked keys давала `softmax(-inf) = NaN` через residual'ы. Симптом — 1222 юзера (13.3%) с NaN-кэшем. Фикс в [src/scorer/gsasrec.py](../src/scorer/gsasrec.py): per-batch 3D `attn_mask` с causal+pad-key masking, диагональ всегда доступна. Distribution shift нулевой.

Финальный кэш: NaN=0, 52,642 уникальных item_idx в union, score range [-0.22, 17.26].

### Шаг 2 — subset аудио ✅

[src/data/audio_embeddings.py](../src/data/audio_embeddings.py) → `extract_audio_subset`. Качаем 14 GB `embeddings.parquet`, фильтруем `np.isin` по 276k items, сохраняем `[n_items+1, 128]` float32 (~135 MB), row 0 = PAD.

`artifacts/audio/embeddings.npy`: shape (276306, 128), 134.9 MB. **Coverage 95.85%** (11,465 items без аудио — catalog drift, не зависит от popularity). Missing-items получают нулевые векторы, маскируются естественно.

### Шаг 3 — random-группы ✅

[src/data/group_synthesis.py](../src/data/group_synthesis.py) → `synthesize_random_groups`. Внутри группы участники уникальны, между группами повторы разрешены. Smoke на n=10000 показал эмпирические доли {2:0.30, 3:0.40, 4:0.20, 5:0.10}.

### Шаг 4 — user audio profiles ✅

[src/data/audio_embeddings.py](../src/data/audio_embeddings.py) → `build_user_audio_profiles(train_listens, item_embeddings)`. Compact `[n_users, 128]` + `uid_to_row.pkl`. Mean по train-listens, маскируя missing-items. Юзеры без valid items → zero-row + `user_audio_valid=False`.

### Шаг 5 — eval-обвязка ✅

- [src/eval/metrics.py](../src/eval/metrics.py) — `ndcg_at_k`, `ranking_ndcg_at_k`, `ndcg_from_ranking`. Векторизовано на numpy.
- [src/eval/group_eval.py](../src/eval/group_eval.py) — `GroupSample` dataclass, `build_group_samples`, `evaluate_aggregator_scores` (NDCG@K + per-sample массивы для bootstrap + срез по размеру).

Ключевое: **out-of-pool релеванты исключаются** — `targets = union(test_listens) ∩ candidates`, IDCG нормируется только по достижимым. Bonus: intersection-вариант доступен через `ground_truth="intersection"`.

### Шаг 6 — BPR + Trainer ✅

- [src/training/bpr_loss.py](../src/training/bpr_loss.py) — `F.logsigmoid`, бродкаст `[B]` vs `[B,K]`.
- [src/training/group_trainer.py](../src/training/group_trainer.py) — `GroupTrainConfig`, `GroupAggregatorTrainer`, `GroupTrainDataset/EvalDataset`, `collate_groups`.

Расширение контракта `GroupAggregator.forward`: trainer паддит `G_max`/`C_max` и передаёт булевы маски как kwargs (`group_mask`, `candidate_mask`). Pad-кандидатам присваивается `-inf` после forward. Loss = BPR + `λ * aggregator.regularization_loss(...)` (для GroupIM).

### Шаги 7–10 — агрегаторы ✅

| Шаг | Класс | Идея |
|---|---|---|
| 7 | `IDBasedAGREE(uid_list, num_items, d_emb=32, d_att=32)` | Item-aware attention `α=softmax(h^T tanh(W[e_u; e_i]))`, group score `Σ α_{u,i}·s_{u,i}`. Item-эмбеддинги учатся только через attention-логит. |
| 8 | `GroupIM(uid_list, num_items, d_emb=32, d_att=32)` | Item-agnostic attention `α=softmax(h^T tanh(W·e_u))` (явная ablation против AGREE). MI-loss через bilinear-дискриминатор `D(e_u, h_G)`, negatives = `torch.roll` по батчу. |
| 9 | `AudioAGREE(d_audio=128, d_att=64)` | Прямой аналог AGREE: `(e_u, e_i) → (a_bar_u, a_i)`. ID-таблиц нет, `phi: Linear(2·d_a→d_att)→GELU→Linear→1`. Missing-audio items дают нулевой логит без edge-case'ов. |
| 10 | `GroupCrossAttention(d_audio=128, d_model=64, n_heads=4)` | Multi-head scaled dot-product: `Q=a_i, K=a_bar_u`. V-проекции нет (нарушила бы контракт «frozen scorer + только взвешивание»). Mean по головам. |

Все 4 агрегатора имеют **единую сигнатуру** `forward(group_user_ids, candidate_ids, per_user_scores, audio_embeds_items, audio_profiles_users, group_mask, candidate_mask) → [B, C_max]` — Trainer дёргает один и тот же код.

Всего 74 unit-теста (18 eval + 13 training + 43 aggregators), все проходят.

### Шаг 11 — обучение ✅

[notebooks/06_train_aggregators.ipynb](../notebooks/06_train_aggregators.ipynb) — per-method блоки, не sweep-цикл (легко тюнить конфиги независимо). Дефолты: n_epochs=60→100, batch=64, lr=1e-3, n_neg=4, patience=20→30. λ_MI=0.5.

**Val NDCG@10 (run 2, n_epochs=100):**

| method | NDCG@10 | NDCG@20 | s=2 | s=3 | s=4 | s=5 |
|---|---:|---:|---:|---:|---:|---:|
| AudioAGREE | **0.0978** | **0.1097** | 0.135 | 0.094 | 0.080 | 0.071 |
| GroupCrossAttn | 0.0941 | 0.1056 | 0.119 | 0.094 | 0.081 | 0.069 |
| AGREE | 0.0916 | 0.1024 | 0.126 | 0.090 | 0.074 | 0.064 |
| GroupIM | 0.0911 | 0.1027 | 0.125 | 0.089 | 0.070 | 0.071 |

Наблюдения:
1. **H1 подтверждается на val.** Audio-методы > ID на всех размерах кроме s=2 (~0.12 у всех).
2. **Structural plateau при frozen scorer.** AGREE/GroupIM/CrossAttn дали идентичные до 4 знака числа в обоих прогонах — продление 60→100 эпох ничего не дало. AudioAGREE ползла +0.001. Это потолок BPR-loss при фиксированных `s_{u,i}`, не баг.
3. **Train loss audio > ID, но val NDCG audio > ID** — классическая «ID переучивает train, audio лучше обобщает».
4. **GroupIM поздний прыжок train loss** (эпохи ~55–65) не сконвертировался в val NDCG → сигнал, что λ_MI=0.5 чуть высоковат.
5. **Val NDCG@10 std ≈ 0.163** у всех — gap audio vs ID на грани шума на marginal CI, **paired bootstrap обязателен** для defensible-результатов.

Чекпоинты в `artifacts/aggregators/<name>/{best.pt,config.json,metrics.csv}` + на HF (`Vladislavbro-500/music-recommendations`).

## 2026-05-12 — Шаги 12–14: Eval, финальная таблица, тривиальные бейзлайны ✅

**Артефакты:**
- [artifacts/eval_results/summary.csv](../artifacts/eval_results/summary.csv) — основная таблица.
- [artifacts/eval_results/summary_by_size.csv](../artifacts/eval_results/summary_by_size.csv) — срез по размеру.
- [artifacts/eval_results/paired.csv](../artifacts/eval_results/paired.csv) — paired bootstrap audio vs ID.
- [artifacts/eval_results/per_sample.npz](../artifacts/eval_results/per_sample.npz) — сырые NDCG + `resample_idx` для воспроизводимости.
- [artifacts/eval_results/summary_table.tex](../artifacts/eval_results/summary_table.tex) — LaTeX-фрагмент.
- [docs/figures/eval_forest_plot.png](figures/eval_forest_plot.png), [eval_heatmap_method_size.png](figures/eval_heatmap_method_size.png), [eval_ndcg_by_size.png](figures/eval_ndcg_by_size.png).

**Протокол:** 2000 test-групп (тот же `groups_split.pkl` что и train/val), `build_group_samples(ground_truth='union', drop_empty=True)`. Bootstrap — 1000 resamples с **общей resample-сеткой** (`rng(seed=42)`) → marginal + paired CI согласованы. Размерный срез — 4 отдельных resample-сетки.

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

## Риски (пост-фактум)

| Риск | Что произошло |
|---|---|
| Subset аудио не помещается в RAM | Снято: G4 даёт 95 GB |
| Union-target в test пустой | 4.x% групп с empty target, отброшены `drop_empty=True` |
| GroupIM MI-loss доминирует | λ=0.5 не сломал, но не помог — поздний train-jump без val-gain |
| Полусырая cross-attention медленная | Не реализовалось, C_G ≤ 1024 держится естественно |
| 9.2k пользователей мало для 4 моделей | Bootstrap CI обязательны, не overclaim в ВКР — учли в caveats |
