"""Prompt A/B plumbing: variants, metadata recording, and run comparison."""
import json

from apm.ai_analyzer import APM_PROMPT_VERSIONS, APMAIAnalyzer, get_system_prompt
from eval.compare_runs import changed_variables
from eval.eval_suite import EvalReport, evaluate_quality_gate, save_eval_report_json


def test_prompt_variants_differ_only_in_grounding_instruction():
    v1, v2 = APM_PROMPT_VERSIONS["v1"], APM_PROMPT_VERSIONS["v2"]
    # Same required sections in both, so an A/B run does not change the output format.
    for section in ("Fault Type", "Root Cause", "Immediate Action", "Prevention", "Severity"):
        assert section in v1
        assert section in v2
    # Only v2 forces claims to cite the metric they rest on.
    assert "cite the metric" in v2
    assert "cite the metric" not in v1


def test_unknown_prompt_version_falls_back_to_baseline():
    assert get_system_prompt("does-not-exist") == APM_PROMPT_VERSIONS["v1"]


def test_analyzer_uses_requested_prompt_version():
    analyzer = APMAIAnalyzer(prompt_version="v2")
    assert analyzer.prompt_version == "v2"
    assert analyzer.system_prompt == APM_PROMPT_VERSIONS["v2"]


def test_report_records_prompt_version(tmp_path):
    report = EvalReport(apm_results=[], log_results=[], overall_score=0.8)
    report.prompt_version = "v2"
    report.quality_gate = evaluate_quality_gate(report)
    data = save_eval_report_json(report, output_path=str(tmp_path / "eval_result.json"))
    assert data["run_metadata"]["prompt_version"] == "v2"


def test_history_filename_separates_prompt_versions(tmp_path):
    for version in ("v1", "v2"):
        report = EvalReport(apm_results=[], log_results=[], overall_score=0.8)
        report.prompt_version = version
        report.quality_gate = evaluate_quality_gate(report)
        save_eval_report_json(report, output_path=str(tmp_path / "eval_result.json"))
    # Same timestamp must not collapse two prompt variants into one file.
    names = sorted(p.name for p in (tmp_path / "history").glob("*.json"))
    assert any("prompt-v1" in n for n in names)
    assert any("prompt-v2" in n for n in names)


def _run(**meta):
    return {"run_metadata": {"dataset_version": "1.1.0", "prompt_version": "v1",
                             "analysis_model": "m", "judge_model": "j",
                             "ollama_version": "0", **meta}}


def test_comparison_flags_a_single_changed_variable():
    assert changed_variables(_run(), _run(prompt_version="v2")) == ["prompt_version"]


def test_comparison_flags_confounded_runs():
    varied = changed_variables(_run(), _run(prompt_version="v2", judge_model="other"))
    assert set(varied) == {"prompt_version", "judge_model"}


def test_comparison_reports_identical_configuration():
    assert changed_variables(_run(), _run()) == []
