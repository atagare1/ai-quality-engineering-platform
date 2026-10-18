"""Fail unless a JUnit report contains executed, non-skipped tests.

Used by live-tests.yml so a skipped or uncollected required live test
cannot produce a green job. Does not print secrets or test payloads.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET


def _cases(path: str) -> list[ET.Element]:
    root = ET.parse(path).getroot()
    found: list[ET.Element] = []
    if root.tag == "testcase":
        found.append(root)
    found.extend(root.findall(".//testcase"))
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("junit_xml")
    parser.add_argument("--min-tests", type=int, default=1)
    args = parser.parse_args()

    try:
        cases = _cases(args.junit_xml)
    except (OSError, ET.ParseError) as exc:
        print(f"ERROR: cannot read JUnit report {args.junit_xml}: {exc}")
        return 1

    skipped = [case for case in cases if case.find("skipped") is not None]
    executed = [case for case in cases if case.find("skipped") is None]
    names = [case.get("name") or case.get("classname") or "?" for case in skipped]

    if len(cases) < args.min_tests:
        print(
            f"ERROR: collected {len(cases)} live test(s); "
            f"expected at least {args.min_tests}"
        )
        return 1
    if skipped:
        print("ERROR: required live test(s) were skipped: " + ", ".join(names))
        return 1
    if len(executed) < args.min_tests:
        print(
            f"ERROR: executed {len(executed)} live test(s); "
            f"expected at least {args.min_tests}"
        )
        return 1
    print(f"OK: {len(executed)} required live test(s) executed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
