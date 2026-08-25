"""
Single-call domain judge.

DeepEval computes its three metrics with roughly nine judge calls per case, each
one a separate extract/verdict/reason round trip. On a CPU-only machine that is
about 280 seconds per case - eight hours for a 102-case set, which means the eval
stops being run at all.

This asks the two remaining questions in one structured call. The third axis -
whether the analysis contradicts the observed metrics - moved to
eval/action_grounding.py after measurement showed neither DeepEval nor an 8B
single-call judge could do it reliably. The scores are built
the same way DeepEval builds them - a ratio of verdicts, counted in code from the
lists the judge returns - so a domain-mode number means the same thing a
deepeval-mode number does, even though it was produced differently. Lists are
asked for rather than counts because a small model counts badly but enumerates
acceptably.

The two modes are not interchangeable evidence: `judge_mode` is recorded in every
report and treated as a changed variable by eval.compare_runs, so a cross-mode
delta is never attributed to the thing under test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class DomainJudgeVerdict(BaseModel):
    """What the judge is required to return. Ollama constrains decoding to this."""

    supported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the analysis that the observed metrics support.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the analysis that the observed metrics neither state nor imply.",
    )
    relevant_statements: list[str] = Field(
        default_factory=list,
        description="Statements that address the requested task.",
    )
    irrelevant_statements: list[str] = Field(
        default_factory=list,
        description="Statements that do not address the requested task.",
    )
    reason: str = Field(default="", description="One or two sentences on the most important finding.")


PROMPT = """You are auditing an incident analysis written by another model.

TASK THE ANALYST WAS GIVEN
{task}

OBSERVED FACTS (the only facts available; treat these as ground truth)
{facts}

THE ANALYSIS TO AUDIT
{answer}

Judge it on two axes and return every field.

1. supported_claims / unsupported_claims - split the analysis's factual claims.
   A claim is supported only when the observed facts state or imply it.
   Reasonable domain inference that the facts do not evidence is unsupported.
2. relevant_statements / irrelevant_statements - split the analysis's statements
   by whether they address the task above. A section the task explicitly asked
   for is relevant even when it is forward looking.

Quote briefly. Do not invent facts that are not listed."""


@dataclass
class DomainScores:
    """Ratio-of-verdicts, built the same way deepeval builds its scores."""

    answer_relevancy: float = -1.0
    faithfulness: float = -1.0
    relevancy_reason: str = ""
    faithfulness_reason: str = ""
    judge_error: str = ""
    verdict: DomainJudgeVerdict | None = field(default=None, repr=False)


def _ratio(good: int, bad: int) -> float:
    """Verdict ratio, or -1.0 when the judge returned nothing to divide."""
    total = good + bad
    return round(good / total, 2) if total else -1.0


def score_verdict(verdict: DomainJudgeVerdict, facts: list[str] | None = None) -> DomainScores:
    faithfulness = _ratio(len(verdict.supported_claims), len(verdict.unsupported_claims))
    relevancy = _ratio(len(verdict.relevant_statements), len(verdict.irrelevant_statements))

    unsupported_note = (
        "Every claim is supported by the metrics."
        if not verdict.unsupported_claims
        else "Unsupported: " + "; ".join(verdict.unsupported_claims[:3])
    )
    irrelevant_note = (
        "Every statement addresses the task."
        if not verdict.irrelevant_statements
        else "Off task: " + "; ".join(verdict.irrelevant_statements[:3])
    )

    return DomainScores(
        answer_relevancy=relevancy,
        faithfulness=faithfulness,
        relevancy_reason=irrelevant_note,
        faithfulness_reason=f"{unsupported_note} {verdict.reason}".strip(),
        verdict=verdict,
    )


def judge(model, task: str, answer: str, facts: list[str]) -> DomainScores:
    """
    One structured call. A judge that cannot honour the schema yields unmeasured
    scores rather than a fabricated number, matching how deepeval mode reports a
    metric it could not compute.
    """
    prompt = PROMPT.format(
        task=task.strip(),
        facts="\n".join(f"- {fact}" for fact in facts),
        answer=answer.strip(),
    )
    try:
        verdict = model.generate(prompt, schema=DomainJudgeVerdict)
    except Exception as exc:
        return DomainScores(judge_error=f"Judge call failed: {type(exc).__name__}: {exc}")

    if not isinstance(verdict, DomainJudgeVerdict):
        return DomainScores(judge_error="Judge did not return the required structure.")
    return score_verdict(verdict, facts)
