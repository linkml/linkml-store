import csv
import json
import sys
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
from pydantic import BaseModel


class Format(Enum):
    JSON = "json"
    JSONL = "jsonl"
    YAML = "yaml"
    TSV = "tsv"
    CSV = "csv"


def load_objects(file_path: Union[str, Path], format: Union[Format, str] = None) -> List[Dict[str, Any]]:
    """
    Load objects from a file in JSON, JSONLines, YAML, CSV, or TSV format.

    :param file_path: The path to the file.
    :param format: The format of the file. Can be a Format enum or a string value.
    :return: A list of dictionaries representing the loaded objects.
    """
    if isinstance(format, str):
        format = Format(format)

    if isinstance(file_path, Path):
        file_path = str(file_path)

    if file_path == "-":
        # set file_path to be a stream from stdin
        f = sys.stdin
    else:
        f = open(file_path)

    if format == Format.JSON or (not format and file_path.endswith(".json")):
        objs = json.load(f)
    elif format == Format.JSONL or (not format and file_path.endswith(".jsonl")):
        objs = [json.loads(line) for line in f]
    elif format == Format.YAML or (not format and (file_path.endswith(".yaml") or file_path.endswith(".yml"))):
        objs = yaml.safe_load(f)
    elif format == Format.TSV or (not format and file_path.endswith(".tsv")):
        reader = csv.DictReader(f, delimiter="\t")
        objs = list(reader)
    elif format == Format.CSV or (not format and file_path.endswith(".csv")):
        reader = csv.DictReader(f)
        objs = list(reader)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    if not isinstance(objs, list):
        objs = [objs]
    return objs


def render_output(data: List[Dict[str, Any]], format: Union[Format, str] = Format.YAML) -> str:
    """
    Render output data in JSON, JSONLines, YAML, CSV, or TSV format.

    :param data: The data to be rendered.
    :param format: The desired output format. Can be a Format enum or a string value.
    :return: The rendered output as a string.
    """
    if isinstance(format, str):
        format = Format(format)

    if isinstance(data, BaseModel):
        data = data.model_dump()

    if format == Format.JSON:
        return json.dumps(data, indent=2, default=str)
    elif format == Format.JSONL:
        return "\n".join(json.dumps(obj) for obj in data)
    elif format == Format.YAML:
        return yaml.safe_dump(data, sort_keys=False)
    elif format == Format.TSV:
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()
    elif format == Format.CSV:
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()
    else:
        raise ValueError(f"Unsupported output format: {format}")
