"""Check data distribution"""

import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import DIFF_QUALITY_DIR

# Check test data
test_file = DIFF_QUALITY_DIR / "cls-test.jsonl"
print(f"Checking {test_file}")

labels = []
with open(test_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 2000:
            break
        data = json.loads(line)
        labels.append(data.get("y", 0))

print(f"Total samples: {len(labels)}")
print(f"Label 0: {labels.count(0)} ({100*labels.count(0)/len(labels):.1f}%)")
print(f"Label 1: {labels.count(1)} ({100*labels.count(1)/len(labels):.1f}%)")

# Check first sample
with open(test_file, "r", encoding="utf-8") as f:
    first = json.loads(f.readline())
    print(f"\nFirst sample keys: {list(first.keys())}")
    print(f"First sample 'y': {first.get('y')}")
    print(f"First sample 'patch' length: {len(first.get('patch', ''))}")
