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

## Measured vs Unmeasured

A metric has no score when every judge call for it failed (for example the judge
model returned invalid JSON). Such a metric is reported as `null` and listed in
`quality_gate.unmeasured_rules` - it is never silently treated as `0.0`.

Classification accuracy skips any case whose answer came from a parser/LLM
fallback (`classification_unmeasured`), because a fallback is not evidence that
the classification was right or wrong.

## Sample Coverage

A metric must be measured on at least `60%` of its applicable cases
(`quality_gate.min_coverage`) before it may certify a pass. A metric that clears
its threshold on thinner data is listed in `insufficient_sample_rules` instead;
`quality_gate.coverage` reports the ratio per metric.

A below-threshold reading always counts as a failure, however thin the sample -
weak evidence of a problem is still evidence.

## Pass / Fail Rule

`quality_gate.status` is one of:

- `PASS` - all 6 metrics measured on sufficient data and at or above threshold.
- `FAIL` - at least one measured metric is below threshold (`failed_rules`).
- `INCONCLUSIVE` - nothing measured below threshold, but at least one metric could
  not be measured (`unmeasured_rules`) or rests on too few cases
  (`insufficient_sample_rules`). Quality is unknown, not bad.

`quality_gate.score` is the fraction of the 6 checks that were certified; failed,
unmeasured, and thin-sample metrics all count as not certified.

## Usage

```bash
py -3 main.py --eval --gate
```

- Exit code `0`: quality gate passed.
- Exit code `2`: quality gate failed (a measured metric is below threshold).
- Exit code `3`: required environment validation failed.
- Exit code `4`: quality gate inconclusive (a metric could not be measured).

## Dataset

Labels live in `eval/datasets/golden_v1.json`, separate from scoring code so they
can be versioned and reviewed on their own. The dataset version is recorded in
every report under `run_metadata.dataset_version`.

Labeling rules:

- Expected types are matched against the parsed classification field, which is
  what the operator actually sees in the alert - not against the whole response.
- Parser fallback values such as `Unknown` are not valid labels.
- Section names the system prompt already mandates (for example `action`) are not
  used as expected keywords, since `completeness_score` covers them.

## Reproducibility

Each report records `run_metadata`: dataset version, analysis model, judge model,
whether the run was self-grading, and the Ollama server version. Every run is also
copied to `reports/history/eval_result_<timestamp>.json` so scores stay comparable
over time.

Set `EVAL_JUDGE_MODEL` to a model other than `OLLAMA_MODEL`; otherwise the model
under test grades its own output and the run is flagged `self_grading: true`.

## Prompt A/B (Regression)

Prompt variants live side by side in `apm/ai_analyzer.py` (`APM_PROMPT_VERSIONS`)
and are selected with `APM_PROMPT_VERSION`:

- `v1` - baseline, no grounding instruction.
- `v2` - every claim must cite the metric it rests on; reasoning about metrics
  absent from the input is forbidden.

The active version is recorded as `run_metadata.prompt_version` and appended to the
history filename, so an A/B pair never overwrites itself:

```bash
APM_PROMPT_VERSION=v1 py -3 main.py --eval
APM_PROMPT_VERSION=v2 py -3 main.py --eval
py -3 -m eval.compare_runs                       # two most recent runs
py -3 -m eval.compare_runs BASE.json NEW.json    # explicit pair
```

`compare_runs` lists which run variables differ and refuses to attribute deltas
when more than one changed. It also names any metric that moved the other way -
a gain in one metric paid for elsewhere is not an improvement.

Attribution caveat: a single pair of runs cannot establish a prompt effect, because
the model is non-deterministic. Repeat each condition and compare distributions
before claiming causality.

## Why This Exists

The quality gate provides explicit QA evidence for:

- repeatable AI quality standards
- regression prevention in CI
- measurable acceptance criteria for SI/SM style quality management
