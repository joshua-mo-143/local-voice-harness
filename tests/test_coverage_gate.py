from __future__ import annotations

import unittest

from local_voice_harness.coverage_gate import coverage_errors


class CoverageGateTests(unittest.TestCase):
    def test_reports_missing_and_undercovered_critical_modules(self) -> None:
        report = {
            "files": {
                "covered.py": {"summary": {"percent_covered": 84.9}},
                "passing.py": {"summary": {"percent_covered": 90}},
            }
        }

        errors = coverage_errors(
            report, {"covered.py": 85, "missing.py": 50, "passing.py": 80}
        )

        self.assertEqual(
            errors,
            [
                "covered.py: 84.9% is below 85.0%",
                "missing.py: missing from coverage report",
            ],
        )

    def test_accepts_modules_at_their_configured_floor(self) -> None:
        report = {"files": {"critical.py": {"summary": {"percent_covered": 70.0}}}}

        self.assertEqual(coverage_errors(report, {"critical.py": 70}), [])


if __name__ == "__main__":
    unittest.main()
