import csv
import json
import sys
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Type, Union

import pandas as pd
import yaml
from pydantic import BaseModel


class Format(Enum):
    """
    Supported generic file formats for loading and rendering objects.
    """

    JSON = "json"
    JSONL = "jsonl"
    YAML = "yaml"
    TSV = "tsv"
    CSV = "csv"
    PARQUET = "parquet"
    FORMATTED = "formatted"


def load_objects(
    file_path: Union[str, Path], format: Union[Format, str] = None, expected_type: Type = None
) -> List[Dict[str, Any]]:
    """
    Load objects from a file in JSON, JSONLines, YAML, CSV, or TSV format.

    >>> load_objects("tests/input/test_data/data.csv")
    [{'id': '1', 'name': 'John', 'age': '30'},
     {'id': '2', 'name': 'Alice', 'age': '25'}, {'id': '3', 'name': 'Bob', 'age': '35'}]

    :param file_path: The path to the file.
    :param format: The format of the file. Can be a Format enum or a string value.
    :param expected_type: The target type to load the objects into.
    :return: A list of dictionaries representing the loaded objects.
    """
    if isinstance(format, str):
        format = Format(format)

    if isinstance(file_path, Path):
        file_path = str(file_path)

    if not format and (file_path.endswith(".parquet") or file_path.endswith(".pq")):
        format = Format.PARQUET

    mode = "r"
    if format == Format.PARQUET:
        mode = "rb"

    if file_path == "-":
        # set file_path to be a stream from stdin
        f = sys.stdin
    else:
        f = open(file_path, mode)

    if format == Format.JSON or (not format and file_path.endswith(".json")):
        objs = json.load(f)
    elif format == Format.JSONL or (not format and file_path.endswith(".jsonl")):
        objs = [json.loads(line) for line in f]
    elif format == Format.YAML or (not format and (file_path.endswith(".yaml") or file_path.endswith(".yml"))):
        if expected_type and expected_type == list:
            objs = list(yaml.safe_load_all(f))
        else:
            objs = yaml.safe_load(f)
    elif format == Format.TSV or (not format and file_path.endswith(".tsv")):
        reader = csv.DictReader(f, delimiter="\t")
        objs = list(reader)
    elif format == Format.CSV or (not format and file_path.endswith(".csv")):
        reader = csv.DictReader(f)
        objs = list(reader)
    elif format == Format.PARQUET:
        import pyarrow.parquet as pq

        table = pq.read_table(f)
        objs = table.to_pandas().to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    if not isinstance(objs, list):
        objs = [objs]
    return objs


def write_output(
    data: Union[List[Dict[str, Any]], Dict[str, Any], pd.DataFrame],
    format: Union[Format, str] = Format.YAML,
    target: Optional[Union[TextIO, str, Path]] = None,
) -> None:
    """
    Write output data to a file in JSON, JSONLines, YAML, CSV, or TSV format.

    >>> write_output([{"a": 1, "b": 2}, {"a": 3, "b": 4}], Format.JSON, sys.stdout)
    [
      {
        "a": 1,
        "b": 2
      },
      {
        "a": 3,
        "b": 4
        }
    ]
    """
    output_str = render_output(data, format)
    if target:
        if isinstance(target, str):
            with open(target, "w") as target:
                target.write(output_str)
        else:
            target.write(output_str)
    else:
        print(output_str)


def render_output(
    data: Union[List[Dict[str, Any]], Dict[str, Any], pd.DataFrame], format: Union[Format, str] = Format.YAML
) -> str:
    """
    Render output data in JSON, JSONLines, YAML, CSV, or TSV format.

    >>> print(render_output([{"a": 1, "b": 2}, {"a": 3, "b": 4}], Format.JSON))
    [
      {
        "a": 1,
        "b": 2
      },
      {
        "a": 3,
        "b": 4
      }
    ]


    :param data: The data to be rendered.
    :param format: The desired output format. Can be a Format enum or a string value.
    :return: The rendered output as a string.
    """
    if isinstance(format, str):
        format = Format(format)

    if format == Format.FORMATTED:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        return str(data)

    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient="records")

    if isinstance(data, BaseModel):
        data = data.model_dump()

    if format == Format.JSON:
        return json.dumps(data, indent=2, default=str)
    elif format == Format.JSONL:
        return "\n".join(json.dumps(obj) for obj in data)
    elif format == Format.YAML:
        if isinstance(data, list):
            return yaml.safe_dump_all(data, sort_keys=False)
        else:
            return yaml.safe_dump(data, sort_keys=False)
    elif format == Format.TSV:
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=get_fieldnames(data), delimiter="\t")
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()
    elif format == Format.CSV:
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=get_fieldnames(data))
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()
    else:
        raise ValueError(f"Unsupported output format: {format}")


def get_fieldnames(data: List[Dict[str, Any]]) -> List[str]:
    """
    Get the fieldnames of a list of dictionaries.

    >>> get_fieldnames([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    ['a', 'b']

    :param data: The list of dictionaries.
    :return: The fieldnames.
    """
    fieldnames = []
    for obj in data:
        fieldnames.extend([k for k in obj.keys() if k not in fieldnames])
    return fieldnames


def guess_format(path: str) -> Optional[Format]:
    """
    Guess the format of a file based on its extension.

    >>> guess_format("data.json")
    <Format.JSON: 'json'>
    >>> guess_format("data.jsonl")
    <Format.JSONL: 'jsonl'>
    >>> guess_format("data.yaml")
    <Format.YAML: 'yaml'>
    >>> assert not guess_format("data")

    :param path: The path to the file.
    :return: The guessed format.
    """
    if path.endswith(".json"):
        return Format.JSON
    elif path.endswith(".jsonl"):
        return Format.JSONL
    elif path.endswith(".yaml") or path.endswith(".yml"):
        return Format.YAML
    elif path.endswith(".tsv"):
        return Format.TSV
    elif path.endswith(".csv"):
        return Format.CSV
    else:
        return None
