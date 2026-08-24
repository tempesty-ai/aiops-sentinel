"""
Action grounding check.

An LLM judge scores whether an answer *reads* correct, not whether it *is*
correct. Code can be verified by running it; an operational recommendation can be
verified by comparing it against the metrics that were actually abnormal.

Observed failure this exists to catch: for a snapshot with cpu=92.5% and
db_connections=12/50, the model answered "Increase database connections to 75".
The DB pool was at 24% - the detector never flagged it - yet HallucinationMetric
scored the answer a perfect 1.0 and the case passed the gate.

The rule is deliberate and narrow: an action may only target a resource the
detector flagged. Grounding is measured against `AnomalyResult.triggered_metrics`,
not against fresh thresholds, so this check cannot drift away from detection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Remediation verbs. A resource mentioned without one of these is treated as
# context or a monitoring note, not as a targeted action.
_REMEDIATION = (
    "increase", "raise", "expand", "extend", "scale", "add", "grow", "enlarge",
    "resize", "bump", "tune", "adjust", "allocate", "provision", "upgrade",
    "증설", "확대", "늘리", "늘려", "상향", "확장", "추가", "조정", "할당",
)

# Resource keys match AnomalyResult.triggered_metrics.
_RESOURCE_PATTERNS: dict[str, tuple[str, ...]] = {
    "db_connection": (
        "db connection", "database connection", "connection pool", "conn pool",
        "pool size", "max_connections", "maxpoolsize", "커넥션", "커넥션 풀", "connection limit",
    ),
    "memory": ("memory", "ram", "메모리"),
    "heap": ("heap", "xmx", "힙"),
    "cpu": ("cpu", "core", "vcpu", "processor", "cpu 코어"),
    "response_time": ("response time", "latency", "timeout", "응답시간", "지연"),
    "error_rate": ("error rate", "error ratio", "에러율", "오류율"),
}

# heap and memory are the same physical pressure from the operator's view, so a
# heap-targeted action is grounded when either was flagged, and vice versa.
_EQUIVALENT: dict[str, tuple[str, ...]] = {
    "memory": ("memory", "heap"),
    "heap": ("heap", "memory"),
}


@dataclass
class GroundingVerdict:
    score: float | None = None            # None = nothing checkable in the action
    targeted: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    reason: str = ""


def _sentences(text: str) -> list[str]:
    return [part for part in re.split(r"[.\n;·]|(?<=다)\s", text or "") if part.strip()]


def find_targeted_resources(action_text: str) -> list[str]:
    """
    Resources the action proposes to change.

    Scoped per sentence so "CPU is high, so increase the DB pool" attributes the
    remediation to the DB pool rather than to every resource in the paragraph.
    """
    targeted: list[str] = []
    for sentence in _sentences(action_text):
        low = sentence.lower()
        if not any(verb in low for verb in _REMEDIATION):
            continue
        for resource, patterns in _RESOURCE_PATTERNS.items():
            if resource in targeted:
                continue
            if any(pattern in low for pattern in patterns):
                targeted.append(resource)
    return targeted


def check_action_grounding(action_text: str, triggered_metrics: list[str]) -> GroundingVerdict:
    """
    Fraction of the action's targeted resources that the detector actually flagged.

    Returns score=None when the action targets no recognizable resource. That is
    reported as unmeasured rather than scored 0.0 - an action can be legitimate
    without naming a metric (for example "collect a heap dump and escalate").
    """
    targeted = find_targeted_resources(action_text)
    if not targeted:
        return GroundingVerdict(score=None, reason="No resource-targeting action to verify.")

    flagged = set(triggered_metrics or [])
    unsupported = [
        resource for resource in targeted
        if not (set(_EQUIVALENT.get(resource, (resource,))) & flagged)
    ]

    score = round((len(targeted) - len(unsupported)) / len(targeted), 2)
    if unsupported:
        reason = (
            f"Action targets {', '.join(unsupported)} but the detector flagged "
            f"{', '.join(sorted(flagged)) or 'nothing'}. Not supported by the metrics."
        )
    else:
        reason = f"Action targets {', '.join(targeted)}, all flagged by the detector."
    return GroundingVerdict(score=score, targeted=targeted, unsupported=unsupported, reason=reason)
