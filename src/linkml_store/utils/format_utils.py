import csv
import json
from enum import Enum
from io import StringIO
from typing import Union, List, Dict, Any

import yaml


class Format(Enum):
    JSON = "json"
    YAML = "yaml"
    TSV = "tsv"
    CSV = "csv"


def load_objects(file_path: str, format: Union[Format, str] = None) -> List[Dict[str, Any]]:
    """
    Load objects from a file in JSON, YAML, CSV, or TSV format.

    :param file_path: The path to the file.
    :param format: The format of the file. Can be a Format enum or a string value.
    :return: A list of dictionaries representing the loaded objects.
    """
    if isinstance(format, str):
        format = Format(format)

    if format == Format.JSON or (not format and file_path.endswith(".json")):
        with open(file_path) as f:
            return json.load(f)
    elif format == Format.YAML or (not format and (file_path.endswith(".yaml") or file_path.endswith(".yml"))):
        with open(file_path) as f:
            return yaml.safe_load(f)
    elif format == Format.TSV or (not format and file_path.endswith(".tsv")):
        with open(file_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            return list(reader)
    elif format == Format.CSV or (not format and file_path.endswith(".csv")):
        with open(file_path) as f:
            reader = csv.DictReader(f)
            return list(reader)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def render_output(data: List[Dict[str, Any]], format: Union[Format, str]) -> str:
    """
    Render output data in JSON, YAML, CSV, or TSV format.

    :param data: The data to be rendered.
    :param format: The desired output format. Can be a Format enum or a string value.
    :return: The rendered output as a string.
    """
    if isinstance(format, str):
        format = Format(format)

    if format == Format.JSON:
        return json.dumps(data, indent=2, default=str)
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