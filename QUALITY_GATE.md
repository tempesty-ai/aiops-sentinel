# QUALITY GATE (AIOps Sentinel)

This document defines the minimum quality bar for AI analysis/classification.

## Gate Metrics

All metrics are measured during `--eval` and stored in `reports/eval_result.json`.

1. `overall_score` >= `0.70`
2. `apm_fault_type_accuracy` >= `0.67`
3. `log_error_type_accuracy` >= `0.50`
4. `hallucination` >= `0.65` (higher is better because score is inverted)
5. `relevancy` >= `0.60`
6. `faithfulness` >= `0.60`

## Pass / Fail Rule

- Pass: all 6 checks pass.
- Fail: one or more checks fail, and failed metric keys are listed in `quality_gate.failed_rules`.

## Usage

```bash
py -3 main.py --eval --gate
```

- Exit code `0`: quality gate passed.
- Exit code `2`: quality gate failed.
- Exit code `3`: required environment validation failed.

## Why This Exists

The quality gate provides explicit QA evidence for:

- repeatable AI quality standards
- regression prevention in CI
- measurable acceptance criteria for SI/SM style quality management
