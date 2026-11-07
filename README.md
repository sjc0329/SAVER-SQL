# SAVER-SQL

Implementation accompanying **SAVER-SQL: Requirement-Guided Diagnostic Evidence for Text-to-SQL Candidate Selection**, Jiancheng Sun.

SAVER-SQL turns question requirements into candidate-specific diagnostic evidence. One question-level LLM call extracts an intent frame; deterministic comparisons derive mismatch indicators for each SQL candidate; an L2-regularized logistic regression selector integrates these indicators with execution consensus, SQL structure and generation provenance. It returns one original SQL from a frozen five-slot pool with a retained evidence record.

## Candidate construction

The upstream strategies are direct generation (C1), alternative sampling (C2), decomposition (C3), query planning (C4), and **conditional query refinement (C5)**. C5 applies a supported deterministic transformation to an executable C1 or uses execution-feedback repair when C1 fails; otherwise it preserves C1. The implementation and archived generation-mode identifiers retain their original names for provenance. All five slots, including duplicates, are retained in the primary experiment.

## Install and run

The reported environment uses Python 3.11.5 and scikit-learn 1.3.0.

```text
python -m pip install -r requirements.txt
python scripts/build_features.py path/to/per_item.jsonl --output outputs/train_features.jsonl
python scripts/train_lr.py outputs/train_features.jsonl --profile full --output-model artifacts/full_lr.pkl
python scripts/select_sql.py --model artifacts/full_lr.pkl --candidates path/to/five_candidate_feature_rows.json
```

Training inputs require candidate-level features and binary training labels. The selection input is a JSON list (or an object with a `candidates` list) of exactly five rows containing `candidate_id`, `generation_mode`, `sql` and `feature_*` values. Profiles are `full`, `lite`, `full_minus_semantic_risk` and `execution_cluster_only`. Training and scoring precomputed features need no API key. Generation and diagnostics use an OpenAI-compatible endpoint configured by `SAVER_API_BASE_URL`, `SAVER_API_KEY` and `SAVER_MODEL`; keep credentials outside version control.

## Frozen-inference companion

The [v1.0.0 frozen-inference companion](https://github.com/sjc0329/SAVER-SQL/releases/tag/v1.0.0) ([download ZIP](https://github.com/sjc0329/SAVER-SQL/releases/download/v1.0.0/SAVER_SQL_Reproducibility_Candidate.zip)) contains four evaluation pools (28,180 candidate slots), raw evaluation features including Spider, archived diagnostics/labels, seven fitted LR models in JSON and original checkpoint formats, anonymous execution-group identifiers, and per-question sensitivity records. The archive preserves the preparation-time README and provenance records; this repository page records its public release status and download link.

After extracting the companion, `python -B replay_release.py` reproduces all four Full+LR evaluations, the three primary LR ablations and Supplementary Table F3 using only Python's standard library. `python -B verify_release.py` verifies its file hashes. The JSON models preserve the training-fitted vocabulary, scales and LR coefficients. Replaying archived inference uses neither benchmark databases nor online LLM calls.

To prepare an input for this repository's selection script:

```text
python scripts/prepare_selection_input.py --root COMPANION --pool bird_deepseek_v4_pro --question-id 31 --profile full_lr --output outputs/q31.json
python scripts/select_sql.py --model COMPANION/models/original_pickles/bird_deepseek_v4_pro__full_lr.pkl --candidates outputs/q31.json
```

Replace `COMPANION` with the companion directory. Use matching adapter and checkpoint profiles for ablations. The adapter joins candidate SQL with features and excludes retrospective labels. Full uses one additional question-level diagnostic call; Lite uses 20 statistical features without that call.

## Results and scope

On the primary BIRD pool, Full+LR answers 1,003/1,534 questions correctly (65.38%), compared with 963 for voting (62.78%). Lite, Full minus semantic risk and execution-cluster-only LR answer 985, 993 and 972 questions correctly. Further source-specific Full+LR evaluations yield 957/1,534 for Qwen3.6-Flash, 1,056/1,534 for Qwen3.6-Plus and 747/1,034 on Spider. Full+LR retains net gains over recomputed voting of 40 questions after C5 removal and 48 after exact-SQL deduplication with models fixed. Diagnostic increments vary with candidate composition and learner.

The code provides feature construction, diagnostics, prompts/rules, fixed LR configuration, training/selection interfaces and archived training-subset identifiers. The companion supports frozen-model inference replay. Training candidate pools, benchmark text/databases, full execution values and specialized-baseline checkpoints are outside the companion. Obtain benchmark data and official evaluators from their original providers for fresh SQL evaluation. The MIT license applies to this implementation; benchmark resources retain their original terms.

## Citation

Jiancheng Sun (2026). SAVER-SQL: Requirement-Guided Diagnostic Evidence for Text-to-SQL Candidate Selection. Manuscript in preparation.
