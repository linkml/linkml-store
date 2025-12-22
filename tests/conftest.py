import csv
import json
import os

import yaml

from tests import INPUT_DIR, OUTPUT_DIR

# Ensure output directory exists for tests that write temp files
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_DATA = [
    {"id": 1, "name": "John", "age": 30},
    {"id": 2, "name": "Alice", "age": 25},
    {"id": 3, "name": "Bob", "age": 35},
]
TEST_DIR = INPUT_DIR / "test_data"
JSON_FILE = os.path.join(TEST_DIR, "data.json")
YAML_FILE = os.path.join(TEST_DIR, "data.yaml")
TSV_FILE = os.path.join(TEST_DIR, "data.tsv")
CSV_FILE = os.path.join(TEST_DIR, "data.csv")

os.makedirs(TEST_DIR, exist_ok=True)
with open(JSON_FILE, "w") as f:
    json.dump(TEST_DATA, f)
with open(YAML_FILE, "w") as f:
    yaml.safe_dump(TEST_DATA, f, sort_keys=False)
with open(TSV_FILE, "w") as f:
    writer = csv.DictWriter(f, fieldnames=TEST_DATA[0].keys(), delimiter="\t")
    writer.writeheader()
    writer.writerows(TEST_DATA)
with open(CSV_FILE, "w") as f:
    writer = csv.DictWriter(f, fieldnames=TEST_DATA[0].keys())
    writer.writeheader()
    writer.writerows(TEST_DATA)
