# EDA Summary — Smart MCQ Solver Challenge

**Milestone 1: NLP Foundation & Semantic Similarity**

## Dataset Overview

| | Train | Test |
|---|---|---|
| Rows | 2,000 | 500 |
| Columns | `id, prompt, A, B, C, D, E, answer` | `id, prompt, A, B, C, D, E` |
| Missing values | 0 | 0 |

Task: given a `prompt` and 5 options (A–E), predict the **top 3 most likely correct answers in ranked order**. Evaluated with **MAP@3**.

---

## 1. Answer Label Distribution

The correct answer is mildly imbalanced across A–E:

| Answer | Count | % |
|---|---|---|
| B | 490 | 24.5% |
| C | 459 | 23.0% |
| A | 369 | 18.4% |
| D | 358 | 17.9% |
| E | 324 | 16.2% |

Not severe enough to require resampling, but worth flagging in error analysis — a model with a slight bias toward predicting B/C first would already align with the base rate.

---

## 2. Duplicate & Near-Duplicate Prompts

- **454 / 2000** train rows (22.7%) share an **exact duplicate prompt** with at least one other row.
- Of the 212 duplicate-prompt groups, only **158 (74.5%)** have fully consistent options + answer — the rest show minor label/content noise, worth a mention in the report's limitations section.
- After stripping common instruction wrappers (e.g. *"Pick the best possible answer:"*, *"...among the listed options."*) to isolate the **core question**, duplication jumps to **1,227 / 2000** train rows — meaning many prompts are the *same underlying question* reworded with a different instruction template.
- **293 core prompts appear in both train and test.** This is the most actionable finding: a meaningful chunk of test questions are near-identical to train questions, just phrased differently. This strongly motivates a **similarity-based / retrieval approach** (TF-IDF or embedding lookup against train) as a high-value strategy, and means train/val splits should be **grouped by core prompt** to avoid leakage.

---

## 3. Text Length Characteristics

**Prompt word count** (median 17, range 3–51):

| Stat | Value |
|---|---|
| Mean | 18.1 |
| Std | 6.8 |
| Min / Max | 3 / 51 |
| 25% / 50% / 75% | 14 / 17 / 22 |

**Option word count** (median 22, much wider spread — options range from 1-word answers to full explanatory sentences):

| Stat | Value |
|---|---|
| Mean | 26.1 |
| Std | 17.2 |
| Min / Max | 1 / 81 |
| 25% / 50% / 75% | 13 / 22 / 37 |

**Train vs test prompt-length distributions overlap closely** — no distribution shift between the two sets, so models validated on a train-derived holdout should generalize to test.

---

## 4. Option Length vs Correctness

- Correct options average **28.7 words**, incorrect options average **25.7 words** — correct options tend to be slightly longer and more detailed.
- **"Always pick the longest option" heuristic scores 34.9% accuracy** (vs. 20% random baseline for top-1). This confirms length is a real, exploitable signal — but explains at most a third of cases, so it's a useful weak feature, not a standalone solution.

---

## 5. Instruction Templates ("Wrapper" Phrasing)

Prompts are built from a small set of fixed instruction templates, e.g.:
- *"Pick the best possible answer: ... among the listed options."*
- *"Select the most accurate option: ... carefully."*
- *"Determine the correct option: ... from the following choices."*
- *"Choose the correct answer: ..."*
- *"Identify the correct statement: ... based on the given context."*

These wrappers carry no semantic signal about the answer and should be **stripped before TF-IDF/embedding similarity** — otherwise they add shared noise across all rows (which slightly inflates apparent similarity between unrelated questions).

---

## 6. Question-Type Breakdown

The dominant question type is **"What"**, followed by "How", "Which", and "Who". Very few "Why"/"When"/"Where" questions. Most content leans toward **factual/definitional recall** (physics, astronomy, philosophy topics observed in sampled rows) rather than reasoning-heavy questions — relevant context for choosing between retrieval-style vs. reasoning-style modeling approaches.

---

## 7. Vocabulary / Content Themes

Word clouds and top n-grams show prompts are dominated by:
- Structural/instructional terms: *given, context, listed, options, following, choices, correct, determine, select*
- Domain content varies widely (physics, astronomy, philosophy terms appear in sampled examples), suggesting the dataset spans **multiple knowledge domains** rather than one narrow subject — reinforces that a general-purpose semantic similarity approach (TF-IDF/embeddings) is more appropriate than domain-specific rules.

---

## 8. Key Takeaways for Modeling (Next Step: TF-IDF Baseline)

1. **Strip instruction wrappers** before vectorizing — keep only the core question + options.
2. **Train/test core-prompt overlap (293 rows)** suggests a similarity-lookup against train could give a strong signal for a meaningful subset of test questions.
3. **Option length is a weak but real signal** (34.9% longest-option accuracy) — consider adding word-count as an auxiliary feature alongside TF-IDF similarity scores.
4. **No missing data, no distribution shift** between train/test — standard TF-IDF cosine-similarity pipeline (prompt vs. each option) is a clean, well-motivated baseline.
5. **Group any train/val split by core prompt**, not exact prompt, to avoid near-duplicate leakage inflating validation scores.

---

*Generated from `01_preprocessing_eda.py` and `02_extensive_eda.py` — see `eda_outputs/` for all supporting plots.*
