"""Tests for linkml-data-browser."""

from pathlib import Path

THIS_DIR = Path(__file__).parent
INPUT_DIR = THIS_DIR / "input"
OUTPUT_DIR = THIS_DIR / "output"

PERSONINFO_SCHEMA = INPUT_DIR / "nested2.schema.yaml"

COUNTRIES_DIR = INPUT_DIR / "countries"
COUNTRIES_CONFIG = COUNTRIES_DIR / "config.yaml"
COUNTRIES_SCHEMA = COUNTRIES_DIR / "countries.linkml.yaml"
COUNTRIES_DB = COUNTRIES_DIR / "countries.db"
COUNTRIES_DATA_JSONL = COUNTRIES_DIR / "countries.jsonl"
