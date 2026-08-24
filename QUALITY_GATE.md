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
7. `action_grounding` >= `0.80`

## Action Grounding

An LLM judge scores whether an answer *reads* correct, not whether it *is*
correct. Code can be checked by running it; an operational recommendation can be
checked against the metrics that were actually abnormal.

The defect this exists for: for a snapshot with `cpu=92.5%` and
`db_connections=12/50`, the model answered *"Increase database connections to
75"*. The pool was at 24% and the detector never flagged it - yet
`HallucinationMetric` scored the answer a perfect `1.0` and the case passed.

`eval/action_grounding.py` extracts the resources an action proposes to change
and requires each one to appear in `AnomalyResult.triggered_metrics`:

```
score = (targeted resources that were flagged) / (targeted resources)
```

Design choices:

- **Measured against detection, not fresh thresholds.** The check reuses what the
  detector already flagged, so it cannot drift away from detection logic.
- **Remediation verbs only.** "CPU is high, so increase the DB pool" targets the
  pool, not the CPU; a resource named without a remediation verb is context.
- **`heap` and `memory` are interchangeable** - the same pressure from an
  operator's point of view.
- **No checkable resource means unmeasured, not 0.0.** "Collect a thread dump and
  escalate" is a legitimate action that names no metric.
- **Deterministic.** This is the half the LLM judge cannot be trusted with, so it
  is plain rules, and it is an independent gate metric rather than an average
  folded into `custom_score` where one bad action would be diluted away.

## Measured vs Unmeasured

A metric has no score when every judge call for it failed (for example the judge
model returned invalid JSON). Such a metric is reported as `null` and listed in
`quality_gate.unmeasured_rules` - it is never silently treated as `0.0`.

Classification accuracy skips any case whose answer came from a parser/LLM
fallback (`classification_unmeasured`), because a fallback is not evidence that
the classification was right or wrong.

## Threshold Resolution

Thresholds are only meaningful if the metric can land near them. With the 5-case
v1 dataset, `apm_fault_type_accuracy` could only take the values
`0 / 0.33 / 0.67 / 1.0`, and its threshold `0.67` sat exactly on that grid - the
rule was really "get 2 of 3 right", and a single case moved `overall_score` by
20 points.

`golden_v2.json` (60 APM / 42 log) puts the resolution at 1/60 for APM accuracy
and under 1 point for `overall_score`, so the thresholds now separate degrees of
quality instead of counting cases. The threshold values themselves are unchanged
and remain judgement calls, not derived numbers.

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

Labels live in `eval/datasets/`, separate from scoring code so they can be
versioned and reviewed on their own. The active file is `golden_v2.json`
(102 cases: 60 APM + 42 log); override with `GOLDEN_DATASET`. The version is
recorded in every report under `run_metadata.dataset_version`.

### Generated from the operational code path

`golden_v2.json` is built by `eval/datasets/build_golden.py`, which drives the
same code the runtime uses:

```
MockAPMGenerator(force_scenario) -> AnomalyDetector -> context_for_ai
```

Two consequences:

- **Labels are correct by construction.** The injected fault scenario *is* the
  ground truth, so 100+ cases need no hand labeling.
- **The eval input has the same shape as the runtime input.** Previously the eval
  used hand-written strings that never passed through the detector, so passing
  the eval said nothing about the distribution the pipeline actually produces.

APM samples the detector does not flag are dropped: the runtime only sends
detected anomalies to the AI, so an unflagged sample is not a valid input.

Rebuild (same seed reproduces the file byte for byte):

```bash
py -3 -m eval.datasets.build_golden --apm 60 --log 42 --seed 20260819
```

### Labeling rules

- A label must be **inferable from the sample alone**. The injector knows it
  produced a `memory_leak`, but one snapshot cannot separate a leak from
  legitimate high usage, so those cases are labeled `memory`/`heap`, never `leak`.
- Expected types are matched against the parsed classification field, which is
  what the operator actually sees in the alert - not against the whole response.
- Parser fallback values such as `Unknown` are not valid labels.
- Section names the system prompt already mandates (for example `action`) are not
  used as expected keywords, since `completeness_score` covers them.
- Labels use word stems so morphological variants both match.

### Phrasing variants

Each log error class carries several real-world phrasings of the *same* fault
(42 log cases, 42 distinct wordings). A classifier that only recognizes one
template scores well on an aggregate while being brittle in practice, so the
report breaks accuracy down per class and per phrasing:

| Scenario | Cases | Accuracy | By phrasing |
| --- | --- | --- | --- |
| connection_refused | 3 | 67% | v1 100%, v2 100%, v3 0% |

The 67% looks like ordinary noise. The breakdown shows it is not: one wording
fails every time. `by_scenario` in the report JSON carries the same data, ordered
weakest first.

APM cases are not varied this way on purpose - their context is produced by
`AnomalyDetector._build_ai_context`, so changing the wording would mean diverging
from the operational format. Their variation comes from the metric values.

### Sampling

A full run costs about 4 LLM calls per case (~408 for 102 cases), which is hours
against a large judge model. `--sample N` evaluates a stratified subset that keeps
every scenario class represented:

```bash
py -3 main.py --eval --sample 20
py -3 main.py --eval --sample 20 --sample-seed 7
```

Sampling is recorded as `cases_evaluated` / `cases_available` / `sample_seed` in
the report, because a score over a subset is not a score over the dataset.

## Reproducibility

Each report records `run_metadata`: dataset version, prompt version, analysis and
judge models **with their content digests**, whether the run was self-grading, the
Ollama server version, and how many cases were evaluated out of how many available.

The digests matter because an Ollama tag is mutable: `ollama pull llama3.1:70b`
can replace the weights behind the same name. Recording only the tag would let two
runs look identically configured while the judge silently changed - the exact false
attribution `compare_runs` exists to prevent. `compare_runs` treats a digest change
as a changed variable and refuses to attribute deltas. Every run is also
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
