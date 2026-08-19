"""
Regression comparison for eval runs.

Compares two stored eval results and reports the delta per metric, so a score
change can be attributed to the one variable that differed (prompt version,
judge model, dataset version) instead of being guessed at.

Usage:
  py -3 -m eval.compare_runs                       # two most recent history files
  py -3 -m eval.compare_runs BASE.json NEW.json    # explicit pair
"""
import glob
import json
import os
import sys

METRIC_ORDER = [
    "overall_score",
    "apm_fault_type_accuracy",
    "log_error_type_accuracy",
    "hallucination",
    "relevancy",
    "faithfulness",
]

META_KEYS = ["dataset_version", "prompt_version", "analysis_model", "judge_model", "ollama_version"]

HISTORY_GLOB = os.path.join("reports", "history", "eval_result_*.json")


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def latest_two() -> tuple[str, str]:
    paths = sorted(glob.glob(HISTORY_GLOB))
    if len(paths) < 2:
        raise SystemExit(f"Need at least 2 history files, found {len(paths)} in {HISTORY_GLOB}")
    return paths[-2], paths[-1]


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def changed_variables(base: dict, new: dict) -> list[str]:
    """Which run variables differ. A clean experiment changes exactly one."""
    bm, nm = base.get("run_metadata", {}), new.get("run_metadata", {})
    return [k for k in META_KEYS if bm.get(k) != nm.get(k)]


def compare(base_path: str, new_path: str) -> dict:
    base, new = load(base_path), load(new_path)
    bg, ng = base.get("quality_gate") or {}, new.get("quality_gate") or {}

    print("=" * 74)
    print("Eval regression comparison")
    print("=" * 74)
    print(f"  BASE  {os.path.basename(base_path)}   ({base.get('timestamp', '?')})")
    print(f"  NEW   {os.path.basename(new_path)}   ({new.get('timestamp', '?')})")

    print("\nRun variables")
    bm, nm = base.get("run_metadata", {}), new.get("run_metadata", {})
    for key in META_KEYS:
        b, n = bm.get(key, "?"), nm.get(key, "?")
        mark = "  <-- changed" if b != n else ""
        print(f"  {key:<18}{str(b):<20}{str(n):<20}{mark}")

    varied = changed_variables(base, new)
    if len(varied) == 1:
        print(f"\n  Controlled: only '{varied[0]}' differs. Deltas are attributable to it.")
    elif not varied:
        print("\n  Identical configuration: deltas reflect LLM run-to-run variance only.")
    else:
        print(f"\n  WARNING: {len(varied)} variables differ ({', '.join(varied)}).")
        print("  Deltas cannot be attributed to any single one.")

    print(f"\n{'metric':<26}{'BASE':>8}{'NEW':>8}{'delta':>9}   samples(base->new)")
    print("-" * 74)
    deltas = {}
    for name in METRIC_ORDER:
        b, n = (bg.get("metrics") or {}).get(name), (ng.get("metrics") or {}).get(name)
        bs = (bg.get("sample_counts") or {}).get(name, "?")
        ns = (ng.get("sample_counts") or {}).get(name, "?")
        if isinstance(b, (int, float)) and isinstance(n, (int, float)):
            delta = round(n - b, 3)
            deltas[name] = delta
            arrow = "+" if delta > 0 else ("" if delta else " ")
            shown = f"{arrow}{delta}"
        else:
            shown = "-"
        print(f"  {name:<24}{_fmt(b):>8}{_fmt(n):>8}{shown:>9}   {bs} -> {ns}")

    print(f"\n  gate status   {bg.get('status', '?')}  ->  {ng.get('status', '?')}")

    print("\nPer-case faithfulness")
    base_cases = {c["case_id"]: c for c in base.get("apm_eval", []) + base.get("log_eval", []) if "case_id" in c}
    new_cases = {c["case_id"]: c for c in new.get("apm_eval", []) + new.get("log_eval", []) if "case_id" in c}
    for cid in sorted(set(base_cases) | set(new_cases)):
        b = base_cases.get(cid, {}).get("faithfulness_score")
        n = new_cases.get(cid, {}).get("faithfulness_score")
        print(f"  {cid:<12}{_fmt(b if isinstance(b, (int, float)) and b >= 0 else None):>8}"
              f"{_fmt(n if isinstance(n, (int, float)) and n >= 0 else None):>8}")

    # A single metric improving is not enough: name what moved the other way.
    regressions = [k for k, v in deltas.items() if v < 0]
    if regressions:
        print(f"\n  Side effects: {', '.join(f'{k} {deltas[k]:+}' for k in regressions)}")
        print("  A gain in one metric paid for elsewhere is not an improvement.")
    else:
        print("\n  No metric regressed.")

    return {"deltas": deltas, "changed_variables": varied}


if __name__ == "__main__":
    if len(sys.argv) == 3:
        compare(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 1:
        compare(*latest_two())
    else:
        raise SystemExit(__doc__)
