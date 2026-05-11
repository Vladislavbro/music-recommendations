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
| 1 | Sanity-чек кэша скоров + выбор/валидация K | заметка в этом логе | ⚠️ (требуется пересчёт кэша, см. ниже) |
| 2 | Subset аудиоэмбеддингов 276k items → `artifacts/audio/embeddings.npy` | numpy + item_id index | ⬜ |
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

### 2026-05-11 — Шаг 1, sanity-чек кэша

**Файл:** `artifacts/user_scores_cache/scores.parquet`, 1,834,000 строк, 11.6 MB.

**Схема (ok):** `uid` int64 (raw user_id YAMBDA, не remap), `item_idx` int64 (0-based, совпадает с `artifacts/gsasrec/item_id_to_idx.pkl`, n_items=276305), `score` float32, `rank` int32.

**K и покрытие:**
- K = 200 фиксированный, ровно 200 строк на каждого uid. Для группового кандидатного пула `C_G = union(top-K members)`, размер групп ≤5 → ≤1000 кандидатов. K достаточный.
- 9170 уникальных uid (из 9194 train post-GTS, 24 потеряны — вероятно эмпти после `filter_listens & played_ratio_pct≥50`).
- 46153 уникальных item_idx во всём union топ-200 (~17% каталога 276k) — ожидаемо (нагрузка на голову популярности).
- item_idx ∈ [1, 276299], 0 (PAD) корректно исключён.
- score-распределение (на ok-юзерах): mean 4.51, std 1.48, min −0.06, max 12.82.

**⚠️ Два связанных бага (оба из-за `InferenceConfig.exclude_history=True`):**

1. **1222 пользователя (13.3%) — весь топ-200 битый:** `score = NaN`, `item_idx = 1..200` (placeholder, появляется когда после маскирования всё стало `-inf` и `torch.topk` возвращает первые K индексов post-PAD). Полностью невалидные строки кэша.
2. **Inconsistency с Phase 1 eval-протоколом.** В Phase 1 финально решили НЕ маскировать историю при ранжировании (закрыло setup-gap 0.0229 → 0.0726, музыкальный домен — переслушивание). Но `cache_user_scores` вызывался с `exclude_history=True` (`src/scorer/inference.py:40,106-108`). Для группового NDCG та же логика → текущий кэш систематически выкидывает кандидатов-переслушивания, которые часть test-таргетов.

**Решение для шага 2 / параллельно:** пересчитать кэш с `exclude_history=False` (одна правка флага в ноутбуке `03_cache_user_scores.ipynb`). На L4 это ~1-2 минуты на 9k user × full-catalog scoring. Альтернатива (отфильтровать 1222 битых uid и оставить mask) — хуже, нарушает eval-протокол Phase 1.

**Открытые подвопросы:**
- Возможно, ещё имеет смысл пересчитать с K=500: aggregator MI-negatives и popularity-negatives могут хотеть бо́льший пул. Отложу решение до шага 6 (group_trainer); если упрёмся — пересчёт дешёвый.

**Статус шага:** диагностика закрыта, пересчёт кэша добавлен как первая задача шага 2 (а не отдельный шаг — слишком мелкий).
