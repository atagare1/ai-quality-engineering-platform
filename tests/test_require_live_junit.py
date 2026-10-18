"""Offline checks for the live-tests JUnit gate."""

from pathlib import Path
import subprocess
import sys

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "require_live_junit.py"
)


def _run(xml_path: Path, min_tests: int = 1) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(xml_path), "--min-tests", str(min_tests)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_require_live_junit_accepts_executed_cases(tmp_path):
    report = tmp_path / "ok.xml"
    report.write_text(
        """<testsuite tests="2" skipped="0">
            <testcase name="a"/><testcase name="b"/>
        </testsuite>""",
        encoding="utf-8",
    )
    completed = _run(report, min_tests=2)
    assert completed.returncode == 0
    assert "OK:" in completed.stdout


def test_require_live_junit_fails_when_required_test_skipped(tmp_path):
    report = tmp_path / "skip.xml"
    report.write_text(
        """<testsuite tests="1" skipped="1">
            <testcase name="live_one"><skipped message="no key"/></testcase>
        </testsuite>""",
        encoding="utf-8",
    )
    completed = _run(report, min_tests=1)
    assert completed.returncode == 1
    assert "skipped" in completed.stdout


def test_require_live_junit_fails_when_nothing_collected(tmp_path):
    report = tmp_path / "empty.xml"
    report.write_text('<testsuite tests="0" skipped="0"/>', encoding="utf-8")
    completed = _run(report, min_tests=1)
    assert completed.returncode == 1
    assert "collected 0" in completed.stdout


def test_require_live_junit_accepts_testsuites_wrapper(tmp_path):
    report = tmp_path / "suites-ok.xml"
    report.write_text(
        """<testsuites>
            <testsuite name="suite-a" tests="1" skipped="0">
                <testcase name="live_a"/>
            </testsuite>
            <testsuite name="suite-b" tests="1" skipped="0">
                <testcase name="live_b"/>
            </testsuite>
        </testsuites>""",
        encoding="utf-8",
    )
    completed = _run(report, min_tests=2)
    assert completed.returncode == 0
    assert "OK:" in completed.stdout


def test_require_live_junit_fails_skipped_inside_testsuites_wrapper(tmp_path):
    report = tmp_path / "suites-skip.xml"
    report.write_text(
        """<testsuites>
            <testsuite name="suite-a" tests="2" skipped="1">
                <testcase name="live_ran"/>
                <testcase name="live_skipped">
                    <skipped message="no key"/>
                </testcase>
            </testsuite>
        </testsuites>""",
        encoding="utf-8",
    )
    completed = _run(report, min_tests=2)
    assert completed.returncode == 1
    assert "skipped" in completed.stdout
    assert "live_skipped" in completed.stdout
