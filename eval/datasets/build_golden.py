"""
Golden-set builder.

Generates a labeled evaluation set from the same code the operational pipeline
uses, so the eval input and the runtime input have identical shape:

    MockAPMGenerator(force_scenario) -> AnomalyDetector -> context_for_ai
    log ERROR template               -> LogTailer-style context

The injected fault scenario is the ground truth, so labels are correct by
construction and no hand labeling is needed.

Labeling rule: a label must be *inferable from the sample alone*. The injector
knows it produced a "memory_leak", but a single snapshot cannot distinguish a
leak from legitimate high usage - a leak needs a trend. So the expected labels
for that scenario are "memory"/"heap", never "leak".

Usage:
  py -3 -m eval.datasets.build_golden                    # default size
  py -3 -m eval.datasets.build_golden --apm 60 --log 40
  py -3 -m eval.datasets.build_golden --seed 7 --out eval/datasets/golden_v3.json
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from apm.anomaly_detector import AnomalyDetector
from apm.mock_generator import MockAPMGenerator
from config.settings import MONITORED_SERVERS

DATASET_VERSION = "2.2.0"

# The runtime context embeds datetime.now(). A dataset must not change every time
# it is rebuilt, so samples get a deterministic stamp derived from their index.
STAMP_BASE = datetime(2026, 1, 1, 0, 0, 0)
DEFAULT_OUT = Path(__file__).parent / "golden_v2.json"

# What a single snapshot of each injected scenario actually lets you conclude.
# Deliberately narrower than the scenario name: see the module docstring.
APM_SCENARIO_LABELS: dict[str, dict[str, list[str]]] = {
    "cpu_spike": {
        "fault_types": ["cpu", "processor", "saturation", "bottleneck"],
        "keywords": ["cpu", "response"],
    },
    "memory_leak": {
        # "leak" is NOT a valid label: not inferable from one point in time.
        "fault_types": ["memory", "heap"],
        "keywords": ["memory", "heap"],
    },
    "db_connection_pool": {
        "fault_types": ["db", "database", "connect", "pool"],
        "keywords": ["db", "connection"],
    },
    "slow_response": {
        "fault_types": ["response", "latency", "slow", "timeout", "performance"],
        "keywords": ["response", "latency"],
    },
    "high_error_rate": {
        "fault_types": ["error", "failure", "exception", "fault"],
        "keywords": ["error", "rate"],
    },
}

# One entry per error class the watcher can surface. Each class carries several
# real-world phrasings of the *same* fault, so the eval measures whether the
# classifier is robust to wording rather than pattern-matching one template.
LOG_TEMPLATES: list[dict] = [
    {
        "key": "connection_refused",
        "error_types": ["network", "connect", "refused"],
        "variants": [
            "ERROR [DataCollector] Connection refused to target host: {host}:{port}",
            "ERROR [DataCollector] java.net.ConnectException: Connection refused - {host}:{port}",
            "ERROR [MetricForwarder] Failed to connect to {host}:{port} after 3 retries (ECONNREFUSED)",
        ],
    },
    {
        "key": "socket_timeout",
        "error_types": ["timeout", "network", "socket"],
        "variants": [
            "ERROR [DataCollector] Socket timeout after {timeout}ms - host: {host}",
            "ERROR [DataCollector] java.net.SocketTimeoutException: Read timed out ({timeout}ms) host={host}",
            "ERROR [HTTPClient] Request to {host} aborted: no response within {timeout}ms",
        ],
    },
    {
        "key": "db_query_failed",
        "error_types": ["db", "database", "query", "sql", "oracle"],
        "variants": [
            "ERROR [DBConnector] Failed to execute query: ORA-{code}: table or view does not exist",
            "ERROR [DBConnector] SQLException while executing statement: ORA-{code} insufficient privileges",
            "ERROR [DBConnector] Query aborted - ORA-{code} deadlock detected while waiting for resource",
        ],
    },
    {
        "key": "queue_overflow",
        "error_types": ["queue", "overflow", "capacity", "forward", "backpressure"],
        "variants": [
            "ERROR [MetricForwarder] Send failed - queue overflow ({queue_size} items pending)",
            "ERROR [MetricForwarder] Dropping metrics: outbound buffer full, {queue_size} entries queued",
            "ERROR [DataBuffer] Backpressure limit exceeded - {queue_size} pending items, rejecting new writes",
        ],
    },
    {
        "key": "heartbeat_lost",
        "error_types": ["agent", "heartbeat", "connect", "availability"],
        "variants": [
            "ERROR [AgentManager] Agent heartbeat lost: agent_id={agent_id}",
            "ERROR [AgentManager] No keepalive from {agent_id} for 90s - marking agent unreachable",
            "ERROR [HealthCheck] Agent {agent_id} missed 3 consecutive heartbeats",
        ],
    },
    {
        "key": "config_permission",
        "error_types": ["permission", "config", "access", "file"],
        "variants": [
            "ERROR [ConfigLoader] Failed to load config file: /etc/collector/config.yml - Permission denied",
            "ERROR [ConfigLoader] java.io.IOException: /opt/agent/settings.conf (Permission denied)",
            "ERROR [ConfigLoader] Cannot read /etc/collector/config.yml - EACCES, running as non-root",
        ],
    },
    {
        "key": "unhandled_exception",
        "error_types": ["exception", "fatal", "thread", "crash"],
        "variants": [
            "FATAL [DataCollector] Unhandled exception in collection thread",
            "FATAL [DataCollector] Uncaught java.lang.NullPointerException in collector-worker-{n}, thread terminated",
            "FATAL [DataCollector] Collection thread died unexpectedly - stack trace unavailable",
        ],
    },
    {
        "key": "http_503",
        "error_types": ["http", "unavailable", "service", "server"],
        "variants": [
            "ERROR [HTTPClient] Request failed with status 503: Service Unavailable",
            "ERROR [HTTPClient] Upstream returned HTTP 503 - retrying with backoff",
            "ERROR [MetricForwarder] Collector endpoint responded 503 Service Unavailable, batch not delivered",
        ],
    },
    {
        "key": "jmx_failed",
        "error_types": ["jmx", "connect", "network", "jvm"],
        "variants": [
            "ERROR [JVMMonitor] JMX connection failed: {host}:{jmx_port}",
            "ERROR [JVMMonitor] Cannot open JMX connector to {host}:{jmx_port} - jmxrmi lookup failed",
            "ERROR [JVMMonitor] MBean server unreachable at {host}:{jmx_port} - JMX metrics unavailable",
        ],
    },
    {
        "key": "malformed_log",
        "error_types": ["parse", "malformed", "format", "log"],
        "variants": [
            "ERROR [LogParser] Malformed log entry - skipping {count} lines",
            "ERROR [LogParser] Unparseable timestamp format, {count} records discarded",
            "ERROR [LogParser] Pattern mismatch on input line - dropped {count} entries this cycle",
        ],
    },
    {
        "key": "out_of_memory",
        "error_types": ["memory", "oom", "outofmemory", "heap"],
        "variants": [
            "Exception in thread collector-worker-{n} java.lang.OutOfMemoryError: Java heap space",
            "FATAL [JVMMonitor] java.lang.OutOfMemoryError: GC overhead limit exceeded",
            "ERROR [DataCollector] Allocation failed - heap exhausted, worker-{n} aborting batch",
        ],
    },
    {
        "key": "ssl_expired",
        "error_types": ["ssl", "certificate", "tls", "security"],
        "variants": [
            "ERROR [SSLHandler] SSL handshake failed: certificate expired",
            "ERROR [SSLHandler] javax.net.ssl.SSLHandshakeException: PKIX path validation failed - cert expired",
            "ERROR [HTTPClient] TLS negotiation with {host} rejected: server certificate no longer valid",
        ],
    },
    {
        "key": "process_hang",
        "error_types": ["process", "hang", "restart", "availability"],
        "variants": [
            "Critical [WatchDog] Process {pid} not responding - triggering restart",
            "Critical [WatchDog] PID {pid} unresponsive for 120s, sending SIGKILL and respawning",
            "ERROR [WatchDog] Liveness probe failed for process {pid} - no progress since last check",
        ],
    },
    {
        "key": "disk_write_failed",
        "error_types": ["disk", "io", "buffer", "write", "storage"],
        "variants": [
            "ERROR [DataBuffer] Buffer flush failed - disk write error: /data/buffer/{n}.buf",
            "ERROR [DataBuffer] java.io.IOException: No space left on device while writing /data/buffer/{n}.buf",
            "ERROR [DataBuffer] Write to /data/buffer/{n}.buf failed (EIO) - spooling to memory",
        ],
    },
]

LOG_MODULES = ["agent_sample_a", "agent_sample_b", "agent_sample_c", "data_forwarder_01", "data_forwarder_02"]


def _fill(line: str, rng: random.Random) -> str:
    return line.format(
        host=f"10.10.{rng.randint(1, 50)}.{rng.randint(1, 200)}",
        port=rng.choice([8080, 8443, 9090, 3306, 1521, 5432]),
        timeout=rng.randint(3000, 30000),
        code=rng.randint(1000, 9999),
        queue_size=rng.randint(500, 5000),
        agent_id=f"agent_{rng.randint(1, 20):02d}",
        jmx_port=rng.randint(9000, 9999),
        count=rng.randint(1, 100),
        n=rng.randint(1, 8),
        pid=rng.randint(10000, 99999),
    )


def build_apm_cases(count: int, rng: random.Random) -> list[dict]:
    """Balanced across scenarios; keeps only samples the detector actually flags."""
    generator = MockAPMGenerator()
    detector = AnomalyDetector()
    scenarios = list(APM_SCENARIO_LABELS)
    cases: list[dict] = []
    attempts = 0

    while len(cases) < count and attempts < count * 40:
        attempts += 1
        scenario = scenarios[len(cases) % len(scenarios)]
        server = rng.choice(MONITORED_SERVERS)
        snapshot = generator.get_snapshot(server, force_scenario=scenario)
        snapshot.timestamp = (STAMP_BASE + timedelta(minutes=len(cases))).strftime("%Y-%m-%d %H:%M:%S")
        result = detector.analyze(snapshot)

        # The runtime pipeline only asks the AI about detected anomalies, so a
        # sample the detector ignores is not a valid eval input.
        if not result.is_anomaly:
            continue

        labels = APM_SCENARIO_LABELS[scenario]
        cases.append(
            {
                "case_id": f"apm-{len(cases) + 1:03d}",
                "name": f"{scenario} on {server}",
                "scenario_key": scenario,
                "server": server,
                "severity_expected": result.severity,
                "triggered_rules": result.triggered_rules,
                "triggered_metrics": result.triggered_metrics,
                "context": result.context_for_ai,
                "expected_keywords": labels["keywords"],
                "expected_fault_types": labels["fault_types"],
            }
        )

    if len(cases) < count:
        raise SystemExit(f"only produced {len(cases)}/{count} APM cases; detector rejected too many samples")
    return cases


def build_log_cases(count: int, rng: random.Random) -> list[dict]:
    """Round-robins the error templates so every class is represented."""
    cases: list[dict] = []
    for i in range(count):
        template = LOG_TEMPLATES[i % len(LOG_TEMPLATES)]
        module = LOG_MODULES[i % len(LOG_MODULES)]
        # Rotate phrasings independently of the class so every variant gets used.
        variant_index = (i // len(LOG_TEMPLATES)) % len(template["variants"])
        line = _fill(template["variants"][variant_index], rng)
        cases.append(
            {
                "case_id": f"log-{i + 1:03d}",
                "name": f"{template['key']} v{variant_index + 1} on {module}",
                "scenario_key": template["key"],
                "variant": variant_index,
                "module": module,
                "context": f"[Log error]\nmodule={module}\nline={line}",
                "expected_error_types": template["error_types"],
            }
        )
    return cases


def build(apm_count: int, log_count: int, seed: int) -> dict:
    rng = random.Random(seed)
    random.seed(seed)  # MockAPMGenerator uses the module-level random
    apm_cases = build_apm_cases(apm_count, rng)
    log_cases = build_log_cases(log_count, rng)

    return {
        "dataset_version": DATASET_VERSION,
        "description": (
            "Golden set generated from the operational code path: MockAPMGenerator with a forced "
            "fault scenario -> AnomalyDetector -> context_for_ai. All data is virtual sample data."
        ),
        "generated_by": "eval/datasets/build_golden.py",
        "seed": seed,
        "labeling_notes": [
            "The injected fault scenario is the ground truth, so labels are correct by construction.",
            "A label must be inferable from the sample alone: 'memory_leak' samples are labeled "
            "memory/heap, never 'leak', because one snapshot cannot separate a leak from high usage.",
            "Expected types are matched against the PARSED classification field, not the whole response.",
            "Parser fallback values such as 'Unknown' are never valid labels.",
            "Labels use word stems so morphological variants both match.",
            "APM samples the anomaly detector does not flag are dropped: the runtime pipeline never "
            "sends those to the AI, so they are not valid eval inputs.",
            "triggered_metrics records which metrics the detector flagged, so the action-grounding "
            "check can verify a recommendation against detection instead of re-deriving thresholds.",
            "Each log error class carries several real-world phrasings of the same fault, so a class "
            "that only passes on one wording shows up as brittleness rather than as a pass.",
        ],
        "apm_cases": apm_cases,
        "log_cases": log_cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the evaluation golden set")
    parser.add_argument("--apm", type=int, default=60, help="number of APM cases")
    parser.add_argument("--log", type=int, default=42, help="number of log cases (multiple of 14 keeps classes even)")
    parser.add_argument("--seed", type=int, default=20260819, help="RNG seed; same seed reproduces the file")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    data = build(args.apm, args.log, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for case in data["apm_cases"] + data["log_cases"]:
        counts[case["scenario_key"]] = counts.get(case["scenario_key"], 0) + 1
    print(f"Wrote {out}  (version {data['dataset_version']}, seed {args.seed})")
    print(f"  APM {len(data['apm_cases'])} / log {len(data['log_cases'])} = {len(data['apm_cases']) + len(data['log_cases'])} cases")
    for key in sorted(counts):
        print(f"    {key:<22}{counts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
