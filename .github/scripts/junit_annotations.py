"""Print GitHub Actions error annotations for failed tests in a JUnit XML report.

Annotations are visible on the run page without downloading logs. Usage:
``python .github/scripts/junit_annotations.py junit.xml``.
"""

import sys
import xml.etree.ElementTree as ET


def escape(text: str) -> str:
    """Escape a message for a workflow command."""
    return text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def main(path: str) -> None:
    """Emit one annotation per failed or erroring test (GitHub shows at most 10 per step)."""
    root = ET.parse(path).getroot()
    failed = 0
    for case in root.iter("testcase"):
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is None:
                continue
            failed += 1
            name = f"{case.get('classname')}::{case.get('name')}"
            details = (node.get("message") or "") + "\n" + (node.text or "")[-1500:]
            print(f"::error title={escape(name)}::{escape(details.strip())}")
    print(f"{failed} failed tests")


if __name__ == "__main__":
    main(sys.argv[1])
