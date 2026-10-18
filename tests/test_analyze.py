import unittest
from datetime import timezone

import analyze


class AnalyzeTests(unittest.TestCase):
    def test_timezone_normalization(self):
        a = analyze.parse_time("2026-09-01T07:00:00+08:00")
        b = analyze.parse_time("2026-08-31T23:00:00Z")
        self.assertEqual(a, b)
        self.assertEqual(a.tzinfo, timezone.utc)

    def test_primary_numbers(self):
        rows = analyze.load_rows(analyze.Path("data/render_tasks.csv"))
        unique = analyze.dedupe_exact(rows)
        self.assertEqual(len(rows), 1127)
        self.assertEqual(len(unique), 1073)
        stats = analyze.stat_block(unique)
        self.assertEqual(stats["failed"], 133)
        self.assertEqual(stats["timeouts"], 45)
        rule = analyze.evaluate_rule(
            "lag>=20", lambda r: r["lag_min"] >= 20, unique, stats["failed"]
        )
        self.assertEqual(rule["flagged_tasks"], 274)
        self.assertEqual(rule["failed_caught"], 100)
        self.assertEqual(rule["net_saved_min"], 6520)

    def test_status_semantics(self):
        rows = analyze.load_rows(analyze.Path("data/render_tasks.csv"))
        self.assertEqual(
            {row["status_code"] for row in rows},
            {"success", "failed", "timeout"},
        )


if __name__ == "__main__":
    unittest.main()
