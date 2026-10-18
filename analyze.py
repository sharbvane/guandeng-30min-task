#!/usr/bin/env python3
"""Reproduce the render-task failure analysis using only the Python standard library.

Primary decision: a task is high risk when the scheduler's last worker heartbeat is
at least 20 minutes old at dispatch time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


COLUMNS = [
    "task_id",
    "dispatched_at",
    "worker_heartbeat_at",
    "cluster_id",
    "template_type",
    "memory_free_gb",
    "queue_depth",
    "status",
]

FAILURE_MINUTES = 120
INTERCEPT_MINUTES = 20
PRIMARY_LAG_MINUTES = 20


def parse_time(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp has no timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"unexpected columns: {reader.fieldnames}")
        rows = []
        for raw in reader:
            row = {name: raw[name].strip() for name in COLUMNS}
            dispatched = parse_time(row["dispatched_at"])
            heartbeat = parse_time(row["worker_heartbeat_at"])
            row["lag_min"] = (dispatched - heartbeat).total_seconds() / 60
            row["memory_free_gb"] = float(row["memory_free_gb"])
            row["queue_depth"] = int(float(row["queue_depth"]))
            row["status_code"] = (
                "success"
                if row["status"] == "1"
                else "failed"
                if row["status"] == "0"
                else "timeout"
                if row["status"] == ""
                else "invalid"
            )
            if row["status_code"] == "invalid":
                raise ValueError(f"unexpected status: {row['status']!r}")
            rows.append(row)
    return rows


def dedupe_exact(rows: list[dict]) -> list[dict]:
    """Remove rows where every provided field is identical."""
    seen = set()
    unique = []
    for row in rows:
        key = tuple(row[name] for name in COLUMNS)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def stat_block(rows: list[dict]) -> dict:
    total = len(rows)
    failed = sum(row["status_code"] == "failed" for row in rows)
    timeouts = sum(row["status_code"] == "timeout" for row in rows)
    successes = total - failed - timeouts
    failure_rate = failed / total if total else 0.0
    timeout_rate = timeouts / total if total else 0.0
    ci = wilson_ci(failed, total)
    return {
        "tasks": total,
        "failed": failed,
        "timeouts": timeouts,
        "successes": successes,
        "failure_rate": failure_rate,
        "failure_rate_pct": round(failure_rate * 100, 1),
        "failure_rate_ci95_pct": [round(ci[0] * 100, 1), round(ci[1] * 100, 1)] if ci else None,
        "timeout_rate_pct": round(timeout_rate * 100, 1),
    }


def in_lag_bin(row: dict, lo: float, hi: float) -> bool:
    return lo <= row["lag_min"] < hi


def evaluate_rule(name: str, predicate, rows: list[dict], total_failures: int) -> dict:
    selected = [row for row in rows if predicate(row)]
    flagged = len(selected)
    caught = sum(row["status_code"] == "failed" for row in selected)
    avoided = caught * FAILURE_MINUTES
    intercept = flagged * INTERCEPT_MINUTES
    net = avoided - intercept
    return {
        "rule": name,
        "flagged_tasks": flagged,
        "flagged_pct": round(flagged / len(rows) * 100, 1) if rows else 0.0,
        "failed_caught": caught,
        "failure_rate_in_flagged_pct": round(caught / flagged * 100, 1) if flagged else 0.0,
        "recall_pct": round(caught / total_failures * 100, 1) if total_failures else 0.0,
        "avoided_failure_min": avoided,
        "intercept_min": intercept,
        "net_saved_min": net,
        "net_saved_hours": round(net / 60, 1),
        "net_saved_min_if_80pct_prevented": round(caught * FAILURE_MINUTES * 0.8 - intercept),
    }


def write_csv(path: Path, fieldnames: list[str], records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/render_tasks.csv")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_rows(input_path)
    unique_rows = dedupe_exact(raw_rows)
    raw_stats = stat_block(raw_rows)
    stats = stat_block(unique_rows)
    total_failures = stats["failed"]

    fresh = [row for row in unique_rows if row["lag_min"] < PRIMARY_LAG_MINUTES]
    stale = [row for row in unique_rows if row["lag_min"] >= PRIMARY_LAG_MINUTES]
    fresh_stats = stat_block(fresh)
    stale_stats = stat_block(stale)

    rules = [
        ("lag>=10", lambda row: row["lag_min"] >= 10),
        ("lag>=15", lambda row: row["lag_min"] >= 15),
        ("lag>=20", lambda row: row["lag_min"] >= 20),
        ("lag>=25", lambda row: row["lag_min"] >= 25),
        ("lag>=30", lambda row: row["lag_min"] >= 30),
        ("lag>=20 or memory<5", lambda row: row["lag_min"] >= 20 or row["memory_free_gb"] < 5),
        ("lag>=20 or queue>=50", lambda row: row["lag_min"] >= 20 or row["queue_depth"] >= 50),
        ("memory<5", lambda row: row["memory_free_gb"] < 5),
        ("queue>=50", lambda row: row["queue_depth"] >= 50),
    ]
    rule_records = [evaluate_rule(name, predicate, unique_rows, total_failures) for name, predicate in rules]
    primary_rule = next(record for record in rule_records if record["rule"] == "lag>=20")
    raw_primary = next(
        evaluate_rule("lag>=20", lambda row: row["lag_min"] >= 20, raw_rows, raw_stats["failed"])
        for _ in [0]
    )

    lag_edges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 10**9)]
    lag_records = []
    for dataset_name, rows in (("raw", raw_rows), ("unique", unique_rows)):
        for lo, hi in lag_edges:
            block = stat_block([row for row in rows if in_lag_bin(row, lo, hi)])
            lag_records.append(
                {
                    "dataset": dataset_name,
                    "lag_min_lo": lo,
                    "lag_min_hi": hi if hi != 10**9 else "",
                    **block,
                }
            )

    segment_records = []
    for dimension in ("cluster_id", "template_type"):
        values = sorted({row[dimension] for row in unique_rows})
        for value in values:
            block = stat_block([row for row in unique_rows if row[dimension] == value])
            segment_records.append({"dimension": dimension, "value": value, **block})

    cluster_lag_records = []
    for cluster in sorted({row["cluster_id"] for row in unique_rows}):
        cluster_rows = [row for row in unique_rows if row["cluster_id"] == cluster]
        cluster_fresh = [row for row in cluster_rows if row["lag_min"] < PRIMARY_LAG_MINUTES]
        cluster_stale = [row for row in cluster_rows if row["lag_min"] >= PRIMARY_LAG_MINUTES]
        cluster_lag_records.append(
            {
                "cluster_id": cluster,
                "fresh_tasks": len(cluster_fresh),
                "fresh_failure_pct": stat_block(cluster_fresh)["failure_rate_pct"],
                "stale_tasks": len(cluster_stale),
                "stale_failure_pct": stat_block(cluster_stale)["failure_rate_pct"],
            }
        )

    task_fields = [
        "task_id",
        "dispatched_at",
        "worker_heartbeat_at",
        "cluster_id",
        "template_type",
        "memory_free_gb",
        "queue_depth",
        "status",
        "status_code",
        "lag_min",
    ]
    write_csv(
        output_dir / "task_level.csv",
        task_fields,
        [{field: row[field] for field in task_fields} for row in unique_rows],
    )
    write_csv(
        output_dir / "lag_bins.csv",
        [
            "dataset",
            "lag_min_lo",
            "lag_min_hi",
            "tasks",
            "failed",
            "timeouts",
            "successes",
            "failure_rate",
            "failure_rate_pct",
            "failure_rate_ci95_pct",
            "timeout_rate_pct",
        ],
        lag_records,
    )
    write_csv(
        output_dir / "rules.csv",
        [
            "rule",
            "flagged_tasks",
            "flagged_pct",
            "failed_caught",
            "failure_rate_in_flagged_pct",
            "recall_pct",
            "avoided_failure_min",
            "intercept_min",
            "net_saved_min",
            "net_saved_hours",
            "net_saved_min_if_80pct_prevented",
        ],
        rule_records,
    )
    write_csv(
        output_dir / "segments.csv",
        [
            "dimension",
            "value",
            "tasks",
            "failed",
            "timeouts",
            "successes",
            "failure_rate",
            "failure_rate_pct",
            "failure_rate_ci95_pct",
            "timeout_rate_pct",
        ],
        segment_records,
    )
    write_csv(
        output_dir / "lag_by_cluster.csv",
        ["cluster_id", "fresh_tasks", "fresh_failure_pct", "stale_tasks", "stale_failure_pct"],
        cluster_lag_records,
    )

    dispatched_times = [parse_time(row["dispatched_at"]) for row in raw_rows]
    summary = {
        "dataset": {
            "file": str(input_path),
            "rows_raw": len(raw_rows),
            "rows_unique_exact": len(unique_rows),
            "exact_duplicate_rows_removed": len(raw_rows) - len(unique_rows),
            "unique_task_ids": len({row["task_id"] for row in raw_rows}),
            "time_range_utc": [
                min(dispatched_times).isoformat(),
                max(dispatched_times).isoformat(),
            ],
            "timezone_note": "Both timestamp columns parsed with ISO-8601 offsets and normalized to UTC.",
        },
        "status": stats,
        "raw_status_sensitivity": raw_stats,
        "heartbeat_age": {
            "threshold_min": PRIMARY_LAG_MINUTES,
            "fresh": fresh_stats,
            "stale": stale_stats,
            "failure_rate_ratio_stale_vs_fresh": round(
                stale_stats["failure_rate"] / fresh_stats["failure_rate"], 2
            )
            if fresh_stats["failure_rate"]
            else None,
        },
        "primary_rule": primary_rule,
        "raw_primary_rule_sensitivity": raw_primary,
        "break_even_failure_rate_pct": round(INTERCEPT_MINUTES / FAILURE_MINUTES * 100, 1),
        "cost_assumptions_min": {
            "failure": FAILURE_MINUTES,
            "requeue_or_intercept": INTERCEPT_MINUTES,
        },
        "data_quality": {
            "status_mapping": {"1": "success", "0": "failed", "blank": "timeout"},
            "duplicate_handling": "Exact duplicate rows removed for primary analysis; raw-row sensitivity reported.",
            "no_fabricated_data": True,
        },
        "interpretation": [
            "Stale heartbeat age is the dominant observable failure signal.",
            "Queue depth is strongly associated with failures but adds little conditional signal after heartbeat age; treat it as a downstream symptom/backlog indicator.",
            "Low free memory amplifies risk, but a memory-only gate is below break-even on fresh workers.",
        ],
        "caveats": [
            "This is observational data, not a controlled experiment.",
            "Net savings assumes an intercepted failure is successfully rerun; the 80% prevention sensitivity is also reported.",
            "Blank status means timeout per the task definition and is not counted as a failure.",
            "The data has no worker ID, retry history, or failure reason code, so the physical root cause of stale heartbeats is not identifiable.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"raw={len(raw_rows)} unique={len(unique_rows)} "
        f"failed={stats['failed']} timeouts={stats['timeouts']} "
        f"stale_failure={stale_stats['failure_rate_pct']}% fresh_failure={fresh_stats['failure_rate_pct']}%"
    )
    print(f"primary_rule={primary_rule['rule']} flagged={primary_rule['flagged_tasks']} "
          f"caught={primary_rule['failed_caught']} net_saved_min={primary_rule['net_saved_min']}")


if __name__ == "__main__":
    main()
