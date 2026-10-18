# SAVER-SQL

## Overview

This repository accompanies *Structured Diagnostic Evidence for Candidate Selection in Text-to-SQL Systems*. SAVER-SQL is a downstream selector that chooses one SQL query from a frozen pool of five candidates without rewriting the selected query. It combines execution behavior, SQL structure, schema grounding, and candidate-specific semantic-risk evidence.

## Method

The upstream pool contains C1-C5. C1 is direct generation, C2 uses sampling, C3 uses decomposition, and C4 uses query planning. C5 is conditional: when C1 executes successfully, a deterministic semantic contrast is attempted; otherwise, an LLM prompt repairs the execution error. It is not an unconditional fifth generation call.

The **Full** representation uses shared question-level diagnostics built only from the question, schema, and external evidence (including conditional value hints), then derives candidate-specific risk indicators. **Lite** uses a fixed set of 20 statistical features and makes no additional diagnostic-LLM call. The primary selector is **Full+Logistic Regression (LR)**, implemented with `DictVectorizer`, `StandardScaler(with_mean=False)`, and L2 logistic regression. Deterministic tie-breaking uses execution-cluster evidence, risk evidence, generation-mode priority, and candidate order. The returned SQL is always one of the original five strings.

## Installation

The reported environment used Python 3.11.5 and scikit-learn 1.3.0.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Training and scoring from precomputed features do not require an API key. Candidate generation and Full diagnostics use an OpenAI-compatible chat-completions endpoint:

```text
SAVER_API_BASE_URL=https://your-provider.example/v1
SAVER_API_KEY=your-key
SAVER_MODEL=your-model-name
```

Do not commit these values. `SAVER_CA_BUNDLE` is optional for a custom CA certificate.

## Usage

Prepare local per-question JSONL records containing the five candidates, their execution metadata, the question-level diagnostic, grounding, and candidate reports. Obtain BIRD data and databases from the [official BIRD benchmark](https://bird-bench.github.io/) and follow its terms. The benchmark data, databases, frozen candidate pool, and SQL text are not included here.

```bash
python scripts/build_features.py path/to/per_item.jsonl --output outputs/train_features.jsonl
python scripts/train_lr.py outputs/train_features.jsonl --profile full --output-model artifacts/full_lr.pkl
python scripts/select_sql.py --model artifacts/full_lr.pkl --candidates path/to/five_candidate_feature_rows.json
```

The selection input is either a JSON list or `{"candidates": [...]}` with exactly five feature rows. Each row must include `candidate_id`, `generation_mode`, `sql`, and the `feature_*` values produced by the feature builder. Other available training profiles are `lite`, `full_minus_semantic_risk`, and `execution_cluster_only`.

## Results

On BIRD dev-full (1,534 questions), Full+LR achieved **65.38%** execution accuracy (1,003/1,534). The LR ablations achieved **64.21%** for Lite+LR, **64.73%** for Full-Semantic Risk+LR, and **63.36%** for Execution-cluster-only+LR. The compact machine-readable summary is in `results/main_results.json`.

## Availability and citation

This repository provides the SAVER-SQL feature construction, diagnostic and grounding logic, authentic C1-C5 prompts/rules, fixed LR configuration, training/scoring interfaces, and an archived training-split record. It does not redistribute BIRD/Spider databases, the frozen candidate SQL pool, raw execution or diagnostic logs, model weights, or implementations of comparison systems. Consequently, the method can be run on user-prepared inputs and the published aggregate arithmetic can be checked, but the complete paper experiment cannot be independently replayed from this repository alone.

Please cite: Fei Chen and Jiancheng Sun (2026). *Structured Diagnostic Evidence for Candidate Selection in Text-to-SQL Systems*. Manuscript in preparation.
