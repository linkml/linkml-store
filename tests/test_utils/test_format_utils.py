import json
import os
import tarfile
import tempfile
from typing import Any, Dict, List, Union

import pytest
import yaml
from linkml_store.utils.format_utils import Format, load_objects, render_output, transform_objects

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


def test_enum():
    # TODO: check handles
    for fmt in Format:
        print(fmt, fmt.is_dump_format())
        assert Format.guess_format(f"foo.{fmt.value}") == fmt
    assert Format.JSON.value == "json"


def test_load_objects_from_tgz():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create test files in different formats
        json_path = os.path.join(tmpdirname, "data.json")
        yaml_path = os.path.join(tmpdirname, "data.yaml")
        tsv_path = os.path.join(tmpdirname, "data.tsv")

        with open(json_path, "w") as f:
            json.dump(TEST_DATA, f)

        with open(yaml_path, "w") as f:
            yaml.safe_dump(TEST_DATA, f)

        with open(tsv_path, "w") as f:
            f.write("id\tname\tage\n")
            for item in TEST_DATA:
                f.write(f"{item['id']}\t{item['name']}\t{item['age']}\n")

        # Create tar.gz archive
        tgz_path = os.path.join(tmpdirname, "data.tar.gz")
        with tarfile.open(tgz_path, "w:gz") as tar:
            tar.add(json_path, arcname="data.json")
            tar.add(yaml_path, arcname="data.yaml")
            tar.add(tsv_path, arcname="data.tsv")

        # Debug: Print contents of tar.gz file
        with tarfile.open(tgz_path, "r:gz") as tar:
            for member in tar.getmembers():
                print(f"File in archive: {member.name}")
                f = tar.extractfile(member)
                if f:
                    content = f.read().decode("utf-8")
                    print(f"Content of {member.name}:")
                    print(content[:100])  # Print first 100 characters
                    print("---")

        # Test load_objects with tgz file
        try:
            loaded_objects = load_objects(tgz_path, compression="tgz")
        except Exception as e:
            print(f"Error in load_objects: {str(e)}")
            raise

        # We expect 3 * len(TEST_DATA) objects because we have 3 files
        assert len(loaded_objects) == 3 * len(TEST_DATA)

        # Check that all original objects are present in the loaded objects
        for original_obj in TEST_DATA:
            matching_objects = [obj for obj in loaded_objects if str(obj["id"]) == str(original_obj["id"])]
            assert len(matching_objects) == 3  # One from each file

            for loaded_obj in matching_objects:
                assert loaded_obj["name"] == original_obj["name"]
                assert int(loaded_obj["age"]) == int(original_obj["age"])  # Convert to int for comparison

OBJS = [
    {
        "id": "P1",
         "address": {
             "street": "1 oak st",
             "city": "Oakland",
         },
     },
     {
        "id": "P2",
         "address": {
             "street": "2 spruce st",
             "city": "Spruceland",
         },
     },
]

@pytest.mark.parametrize(
    "objects, select_expr, expected",
    [
        ([], None, []),
        ([], "x", []),
        (OBJS, None, OBJS),
        (OBJS, "id", ["P1", "P2"]),
        (OBJS, "address.city", ["Oakland", "Spruceland"]),
]
)
def test_transform_objects(objects, select_expr, expected):
    tr_objects = transform_objects(objects, select_expr)
    assert tr_objects == expected
