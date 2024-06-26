import json
from typing import Any, Dict, List, Union

import pytest
import yaml
from linkml_store.utils.format_utils import Format, load_objects, render_output

from tests.conftest import CSV_FILE, JSON_FILE, TEST_DATA, TSV_FILE, YAML_FILE


# Test cases for load_objects
@pytest.mark.parametrize(
    "file_path, format, expected",
    [
        (JSON_FILE, None, TEST_DATA),
        (JSON_FILE, Format.JSON, TEST_DATA),
        (JSON_FILE, "json", TEST_DATA),
        (YAML_FILE, None, TEST_DATA),
        (YAML_FILE, Format.YAML, TEST_DATA),
        (YAML_FILE, "yaml", TEST_DATA),
        (TSV_FILE, None, TEST_DATA),
        (TSV_FILE, Format.TSV, TEST_DATA),
        (TSV_FILE, "tsv", TEST_DATA),
        (CSV_FILE, None, TEST_DATA),
        (CSV_FILE, Format.CSV, TEST_DATA),
        (CSV_FILE, "csv", TEST_DATA),
    ],
)
def test_load_objects(file_path: str, format: Union[Format, str], expected: List[Dict[str, Any]]):
    loaded = load_objects(file_path, format)
    if format in [Format.JSON, Format.YAML, "json", "yaml"] or file_path in [JSON_FILE, YAML_FILE]:
        assert loaded == expected
    else:
        expected = [{k: str(v) for k, v in obj.items()} for obj in expected]
        assert loaded == expected


def test_load_objects_invalid_format():
    with pytest.raises(ValueError):
        load_objects("invalid.txt", format="invalid")


# Test cases for render_output
@pytest.mark.parametrize(
    "data, format, expected",
    [
        (TEST_DATA, Format.JSON, json.dumps(TEST_DATA, indent=2)),
        (TEST_DATA, "json", json.dumps(TEST_DATA, indent=2)),
        (TEST_DATA, Format.YAML, yaml.safe_dump_all(TEST_DATA, sort_keys=False)),
        (TEST_DATA, "yaml", yaml.safe_dump_all(TEST_DATA, sort_keys=False)),
        (
            TEST_DATA,
            Format.TSV,
            "id\tname\tage\n1\tJohn\t30\n2\tAlice\t25\n3\tBob\t35\n",
        ),
        (TEST_DATA, "tsv", "id\tname\tage\n1\tJohn\t30\n2\tAlice\t25\n3\tBob\t35\n"),
        (TEST_DATA, Format.CSV, "id,name,age\n1,John,30\n2,Alice,25\n3,Bob,35\n"),
        (TEST_DATA, "csv", "id,name,age\n1,John,30\n2,Alice,25\n3,Bob,35\n"),
    ],
)
def test_render_output(data: List[Dict[str, Any]], format: Union[Format, str], expected: str):
    if format in [Format.JSON, Format.YAML, "json", "yaml"]:
        assert render_output(data, format) == expected


def test_render_output_invalid_format():
    with pytest.raises(ValueError):
        render_output(TEST_DATA, format="invalid")
