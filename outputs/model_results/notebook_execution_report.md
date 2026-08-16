# Notebook execution report

Execution date: 2026-08-16 (UTC)

All affected notebooks were run from the first code cell through the final export cell using the pinned project environment. The hosted sandbox blocks the local TCP sockets used by Jupyter kernels, so execution used an in-process notebook runner with the same shared Python namespace and working directory semantics. Every non-empty code cell completed, received an execution count, and has no error output.

## End-to-end status

| Notebook | Result | Runtime |
|---|---:|---:|
| `03_response_baseline.ipynb` | PASS | 0.30 min |
| `04_t_learner.ipynb` | PASS | 4.26 min |
| `05_s_learner.ipynb` | PASS | 0.36 min |
| `06_x_learner.ipynb` | PASS | 3.98 min |
| `07_r_learner.ipynb` | PASS | 2.74 min |
| `causal_forest.ipynb` | PASS | 57.83 min |
| `10_evaluation_qini_auuc.ipynb` | PASS | 0.12 min |

The forest runtime includes five 200-tree out-of-fold fits followed by one final 200-tree fit on all 48,000 training rows.

## Validation and held-out test results

Within each family, the variant with the highest five-fold out-of-fold validation Qini was selected before its untouched-test result was read.

| Family | Validation-selected model | Validation Qini | Test Qini | Validation AUUC | Test AUUC |
|---|---|---:|---:|---:|---:|
| Response Baseline | Response Baseline (XGB) | 0.040052 | 0.062611 | 0.537346 | 0.565292 |
| T-Learner | T-Learner (XGB) | 0.080797 | 0.052379 | 0.579523 | 0.554409 |
| X-Learner | X-Learner | 0.059326 | 0.050503 | 0.558244 | 0.551560 |
| R-Learner | R-Learner | 0.068731 | 0.045817 | 0.566783 | 0.547063 |
| S-Learner | S-Learner (XGB) | 0.063511 | 0.038087 | 0.560544 | 0.541280 |
| Uplift Random Forest | Uplift Random Forest | 0.048572 | 0.006527 | 0.547398 | 0.507892 |

The response baseline has the highest test Qini in this one split, but it was not selected by looking at test performance. The T-Learner XGB variant has the highest validation Qini. The uplift random forest has the largest validation-to-test degradation (Qini gap 0.042045).

## S-Learner comparability check

The S-Learner now uses the same normalized CausalML Qini and AUUC implementation, treatment-allocation adjustment, OOF selection protocol, and decile direction as the other learners.

| S-Learner variant | Validation Qini | Test Qini | Validation AUUC | Test AUUC |
|---|---:|---:|---:|---:|
| Logistic regression | 0.033034 | 0.069852 | 0.531526 | 0.573516 |
| XGBoost | 0.063511 | 0.038087 | 0.560544 | 0.541280 |

XGBoost is correctly selected because it wins on training-only OOF validation. Logistic regression happens to score better on the held-out test, but switching to it after seeing test performance would reintroduce test leakage.

## Prediction fingerprint audit

The execution generated six 16,000-row test prediction files and six 48,000-row OOF prediction files. Every file passed all of these checks:

- canonical sequential `test_row_id` or `train_row_id`;
- exact `y_true` equality with the canonical split;
- exact `treatment` equality with the canonical split;
- expected row count;
- no missing values.

This includes `tau_hat_x.csv` and `tau_hat_r.csv`, which now contain `test_row_id`, `y_true`, and `treatment` instead of an unjoinable saved DataFrame index.

The row-level prediction CSVs are published with this report under the project owner's explicit authorization, including individual outcomes, treatment assignments, and model scores. The generated Qini, AUUC, decile, propensity, and forest comparison figures were visually inspected in the execution environment; their numeric source results are captured in the published CSV and JSON artifacts.

## Runtime defects found and fixed

- T-Learner grid searches used `n_jobs=-1`, which failed in the constrained runtime; the notebook now uses deterministic single-process fitting.
- T-Learner passed an unsupported `ax=` argument to CausalML's installed `plot_qini`; it now uses the project's shared plotting helpers.
- CausalML 0.15.5 hard-codes process-based `n_jobs=-1` inside `BaseRClassifier` cross-validation; R-Learner now executes those fits under Joblib's thread backend.
- The unconstrained resolver selected Numba/LLVM versions that caused a CausalML import bus error; execution used compatible `numba==0.60.0` and `llvmlite==0.43.0` pins.

## Saved review artifacts

- `validation_test_uplift_comparison.csv`
- `model_variant_validation_scores.csv`
- per-model JSON summaries under `outputs/model_results/`
- per-model JSON summaries, including decile, segment, and ROI diagnostics where applicable
