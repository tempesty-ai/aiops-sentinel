# QUALITY GATE (AIOps Sentinel)

This document defines the minimum quality bar for AI analysis/classification.

## Gate Metrics

All metrics are measured during `--eval` and stored in `reports/eval_result.json`.

1. `overall_score` >= `0.70`
2. `apm_fault_type_accuracy` >= `0.67`
3. `log_error_type_accuracy` >= `0.50`
4. `cause_grounding` >= `0.80`
5. `action_grounding` >= `0.80`
6. `relevancy` >= `0.60`
7. `faithfulness` >= `0.60`

Metrics 1-5 are computed by rules; 6 and 7 by an LLM judge. Which check owns which
axis was decided by measurement, not preference - see *Who grades what*.

## Grounding: cause and action

An LLM judge scores whether an answer *reads* correct, not whether it *is*
correct. Code can be checked by running it; an operational diagnosis or
recommendation can be checked against the metrics that were actually abnormal.

Two observed defects this exists for:

| Snapshot | Model answered | Reality |
| --- | --- | --- |
| `cpu=92.5%`, `db_connections=12/50` | "Increase database connections to 75" | the pool was at 24% and never flagged |
| `db_connections=47/50`, `cpu=32%` | "Scale up server resources (CPU, Memory)" | CPU and memory were both normal |

`HallucinationMetric` scored the first a perfect `1.0`.

`eval/action_grounding.py` extracts the resources the answer points at and
requires each to appear in `AnomalyResult.triggered_metrics`:

```
score = (pointed-at resources that were flagged) / (pointed-at resources)
```

The two axes are graded separately because the fixes differ - a wrong diagnosis
is a reasoning problem, a wrong remedy is a recommendation problem, and an
aggregate would hide which one happened:

- `cause_grounding` reads the **Root Cause** section. Any resource named there is
  being blamed, so no verb is required.
- `action_grounding` reads the **Immediate Action** section and requires a
  remediation verb. "CPU is high, so increase the DB pool" targets the pool.

Other design choices:

- **Measured against detection, not fresh thresholds.** The check reuses what the
  detector already flagged, so it cannot drift away from detection logic.
- **`heap` and `memory` are interchangeable** - the same pressure from an
  operator's point of view.
- **No checkable resource means unmeasured, not 0.0.** "Collect a thread dump and
  escalate" is a legitimate action that names no metric.
- **Deterministic**, and an independent gate metric rather than a component of
  `custom_score` where one bad answer would be diluted away.

## Who grades what

`hallucination` used to be a gate metric. It is not any more, and the reason is
measured rather than argued:

| Grader | Result on the "contradicts the metrics" axis |
| --- | --- |
| DeepEval `HallucinationMetric`, 70B judge | `1.0` on all 5 cases, including a clear contradiction. No discrimination. |
| Single-call 8B judge, metrics unlabelled | Read a 94.1% reading against an 80% threshold as "within normal range". |
| Single-call 8B judge, metrics pre-labelled | Still wrong on 2 of 3 cases; flagged facts it then described as correct. |
| Rules (`cause_grounding`) | Correct on every case tried, with the specific resource named. |

So the axis moved to rules, and the LLM judge kept only what rules cannot express:

- **Rules own** classification accuracy, keyword coverage, completeness, and both
  grounding axes.
- **The judge owns** `faithfulness` (are the claims evidenced) and `relevancy`
  (does the answer address the task).

An 8B judge on CPU is the practical ceiling here, so the split is drawn where that
judge is actually reliable.

## Hallucination rate (reported, not gated)

The gate has no `hallucination` metric, but the report carries a hallucination
figure so the concept stays readable:

```
Hallucination rate: 33% - 20 of 60 cases asserted something the metrics do not
support (diagnosis 14, action 11). Most often: cpu x18, memory x9
```

It is **derived** from `cause_grounding` and `action_grounding` - the two places
this project can detect a hallucination deterministically - and deliberately not
gated, because the gate already holds both axes to `0.80` and gating the roll-up
would count the same evidence twice.

Reported as a *rate* so the direction is unambiguous: `0.0` means nothing
unsupported. The removed metric was inverted, read backwards from its own name,
and scored `1.0` on an answer that contradicted the data outright.

A case whose answer names no checkable resource is left out of the denominator -
saying nothing specific is not evidence of honesty.

## Judge modes

```bash
py -3 main.py --eval           # domain judge: 1 call per case  (~1.8h for 102 cases)
py -3 main.py --eval --deep    # DeepEval:     9 calls per case (~8h for 102 cases)
```

DeepEval splits every metric into separate extract / verdict / reason round trips.
That is a sound trade against a hosted model at ~0.3s per call; against a local 8B
model on CPU at ~20s per call it is 918 calls and eight hours, which means the
eval stops being run at all.

The domain judge asks the two remaining questions in one structured call. Scores
are still ratios of verdicts, counted in code from the lists the judge returns, so
a number means the same thing in both modes.

`judge_mode` is recorded in every report and treated as a changed variable by
`compare_runs`, so a cross-mode delta is never attributed to the thing under test.

## Structured output

The judge adapter implements `generate(prompt, schema=...)`, which lets DeepEval
hand it a schema and lets Ollama constrain decoding to it.

Without this, DeepEval's `generate_with_schema` hit a `TypeError`, fell back to
free-text parsing, and every judge below 70B failed with *"Evaluation LLM
outputted an invalid JSON"* - the metrics were unmeasurable for a reason that had
nothing to do with the models. With it, an 8B judge measures all of them.

## Fact granularity

`HallucinationMetric` emits one verdict per context document, so passing the whole
snapshot as a single blob capped the score at `0.0` or `1.0`. The context is now
split per metric line, and each line is tagged with the detector's own verdict
(`[ABNORMAL]` / `[NORMAL]`) so the judge never has to compare a value against a
threshold - it got that wrong in both directions.

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
(102 cases: 60 APM + 42 log); override with `GOLDEN_DATASET`. It is the only dataset
shipped - the retired 5-case set carried a label a single snapshot cannot support
("leak"), and leaving it in the tree invited an eval that looks valid and is not.
Git holds the history, and the builder reproduces any size from a seed. The version is
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
