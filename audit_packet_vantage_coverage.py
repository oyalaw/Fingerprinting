from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    cmd = [
        sys.executable,
        str(ROOT / "build_packet_cross_vantage_dataset.py"),
        "--audit-only",
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    audit = (
        ROOT
        / "fingerprinting_dataset"
        / "packet_cross_vantage"
        / "source_coverage_audit.json"
    )
    if not audit.exists():
        return
    data = json.loads(audit.read_text(encoding="utf-8"))
    print("\nAudit file:", audit)
    skipped = data.get("skipped", [])
    if skipped:
        reasons = Counter(item.get("reason", "unknown") for item in skipped)
        print("Skipped/unresolved items:", len(skipped))
        for reason, count in reasons.most_common():
            print(f"  {count:4d}  {reason}")
    else:
        print("Skipped/unresolved items: 0")


if __name__ == "__main__":
    main()
