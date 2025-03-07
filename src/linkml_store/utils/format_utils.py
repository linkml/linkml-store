import csv
import gzip
import hashlib
import io
import json
import logging
import re
import sys
import tarfile
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, TextIO, Type, Union

import pandas as pd
import pystow
import xmltodict
import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Format(Enum):
    """
    Supported generic file formats for loading and rendering objects.
    """

    JSON = "json"
    JSONL = "jsonl"
    YAML = "yaml"
    YAMLL = "yamll"
    TOML = "toml"
    TSV = "tsv"
    CSV = "csv"
    XML = "xml"
    TURTLE = "turtle"
    RDFXML = "rdfxml"
    TEXT = "text"
    TEXTLINES = "textlines"
    OBO = "obo"
    FASTA = "fasta"
    GMT = "gmt"
    DAT = "dat"
    MARKDOWN = "markdown"
    PKL = "pkl"
    PYTHON = "python"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    NETCDF = "netcdf"
    FORMATTED = "formatted"
    TABLE = "table"
    XLSX = "xlsx"
    SQLDUMP_DUCKDB = "duckdb"
    SQLDUMP_POSTGRES = "postgres"
    DUMP_MONGODB = "mongodb"

    @classmethod
    def guess_format(cls, file_name: str) -> Optional["Format"]:
        ext = Path(file_name).suffix.lower()

        format_map = {
            ".json": cls.JSON,
            ".jsonl": cls.JSONL,
            ".yaml": cls.YAML,
            ".yml": cls.YAML,
            ".yamll": cls.YAMLL,
            ".tsv": cls.TSV,
            ".csv": cls.CSV,
            ".txt": cls.TEXT,
            ".xml": cls.XML,
            ".owx": cls.XML,
            ".owl": cls.RDFXML,
            ".ttl": cls.TURTLE,
            ".md": cls.MARKDOWN,
            ".py": cls.PYTHON,
            ".parquet": cls.PARQUET,
            ".pq": cls.PARQUET,
        }
        fmt = format_map.get(ext, None)
        if fmt is None:
            if ext.startswith("."):
                ext = ext[1:]
            if ext in [f.value for f in Format]:
                return Format(ext)
        return fmt

    def is_dump_format(self):
        return self in [Format.SQLDUMP_DUCKDB, Format.SQLDUMP_POSTGRES, Format.DUMP_MONGODB]

    def is_binary_format(self):
        return self in [Format.PARQUET, Format.XLSX]

    def is_xsv(self):
        return self in [Format.TSV, Format.CSV]


def load_objects_from_url(
    url: str,
    format: Union[Format, str] = None,
    expected_type: Type = None,
    local_path: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Load objects from a URL in JSON, JSONLines, YAML, CSV, or TSV format.

    :param url: The URL to the file.
    :param format: The format of the file. Can be a Format enum or a string value.
    :param expected_type: The target type to load the objects into.
    :param local_path: The local path to save the file to.
    :return: A list of dictionaries representing the loaded objects.
    """
    local_path = pystow.ensure("linkml", "linkml-store", url=url)
    logger.info(f"synced to {local_path}")
    objs = load_objects(local_path, format=format, expected_type=expected_type, **kwargs)
    if not objs:
        raise ValueError(f"No objects loaded from URL: {url}")
    return objs


def clean_pandas_value(v):
    """Clean a single value from pandas."""
    import math

    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)  # Ensures proper float type
    return v


def clean_nested_structure(obj):
    """Recursively clean a nested structure of dicts/lists from pandas."""
    if isinstance(obj, dict):
        return {k: clean_nested_structure(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nested_structure(item) for item in obj]  # Fixed: using 'item' instead of 'v'
    else:
        return clean_pandas_value(obj)


def process_file(
    f: IO,
    format: Format,
    expected_type: Optional[Type] = None,
    header_comment_token: Optional[str] = None,
    format_options: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Process a single file and return a list of objects.

    :param f: The file object.
    :param format: The format of the file.
    :param expected_type: The expected type of the objects.
    :param header_comment_token: Token used for header comments to be skipped
    :return:
    """
    if format_options is None:
        format_options = {}
    if format == Format.YAMLL:
        format = Format.YAML
        expected_type = list
    if format == Format.JSON:
        objs = json.load(f)
    elif format == Format.JSONL:
        objs = [json.loads(line) for line in f]
    elif format == Format.YAML:
        if expected_type and expected_type == list:  # noqa E721
            objs = list(yaml.safe_load_all(f))
            # allow YAML with a `---` with no object before it
            objs = [obj for obj in objs if obj is not None]
        else:
            objs = yaml.safe_load(f)
    elif format == Format.TOML:
        import toml

        objs = toml.load(f)
        if not isinstance(objs, list):
            objs = [objs]
    elif format == Format.TEXTLINES:
        objs = f.readlines()
    elif format in [Format.TSV, Format.CSV]:
        if header_comment_token:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line.startswith(header_comment_token):
                    f.seek(pos)
                    break
        delimiter = "\t" if format == Format.TSV else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        objs = list(reader)
    elif format == Format.XML:
        objs = xmltodict.parse(f.read())
    elif format == Format.PKL:
        objs = pd.read_pickle(f).to_dict(orient="records")
    elif format == Format.XLSX:
        xls = pd.ExcelFile(f)
        objs = {sheet: clean_nested_structure(xls.parse(sheet).to_dict(orient="records")) for sheet in xls.sheet_names}
    elif format == Format.TEXT:
        txt = f.read()
        objs = [
            {
                "name": Path(f.name).name,
                "path": f.name,
                "content": txt,
                "size": len(txt),
                "lines": txt.count("\n") + 1,
                "md5": hashlib.md5(txt.encode()).hexdigest(),
            }
        ]
    elif format == Format.GMT:
        objs = []
        lib_name = Path(f.name).name
        for line in f:
            parts = line.strip().split("\t")
            desc = parts[1]
            objs.append(
                {
                    "library": lib_name,
                    "uid": f"{lib_name}.{parts[0]}",
                    "name": parts[0],
                    "description": desc if desc else None,
                    "genes": parts[2:],
                }
            )
    elif format == Format.FASTA:
        objs = []
        current_obj = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_obj:
                    objs.append(current_obj)
                current_obj = {"id": line[1:], "sequence": ""}
            else:
                current_obj["sequence"] += line
        if current_obj:
            objs.append(current_obj)
    elif format == Format.OBO:
        blocks = split_document(f.read(), "\n\n")
        id_pattern = re.compile(r"id: (\S+)")

        def get_id(block):
            m = id_pattern.search(block)
            return m.group(1) if m else None

        objs = [{"id": get_id(block), "content": block} for block in blocks]
        objs = [obj for obj in objs if obj["id"]]
    elif format == Format.DAT:
        from linkml_store.utils.dat_parser import parse_sib_format

        _, objs = parse_sib_format(f.read())
    elif format in (Format.RDFXML, Format.TURTLE):
        import lightrdf

        parser = lightrdf.Parser()
        objs = []
        ext_fmt = "rdfxml"
        if format == Format.TURTLE:
            ext_fmt = "ttl"
        bytesio = io.BytesIO(f.read().encode("utf-8"))
        buffer = io.BufferedReader(bytesio)
        for s, p, o in parser.parse(buffer, base_iri=None, format=ext_fmt):
            obj = {
                "subject": s,
                "predicate": p,
                "object": o,
            }
            if format_options.get("pivot", False):
                obj = {
                    "subject": s,
                    p: o,
                }
            objs.append(obj)
    elif format == Format.PARQUET:
        import pyarrow.parquet as pq

        table = pq.read_table(f)
        objs = table.to_pandas().to_dict(orient="records")
    elif format in [Format.PYTHON, Format.FORMATTED, Format.TABLE]:
        raise ValueError(f"Format {format} is not supported for loading objects")
    else:
        raise ValueError(f"Unsupported file format: {format}")

    if not isinstance(objs, list):
        objs = [objs]
    return objs


def load_objects(
    file_path: Union[str, Path],
    format: Optional[Union[Format, str]] = None,
    compression: Optional[str] = None,
    expected_type: Optional[Type] = None,
    header_comment_token: Optional[str] = None,
    select_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load objects from a file or archive in supported formats.
    For tgz archives, it processes all files and concatenates the results.

    TODO: Add schema hints for CSV/TSV parsing.

    :param file_path: The path to the file or archive.
    :param format: The format of the file. Can be a Format enum or a string value.
    :param compression: The compression type. Supports 'gz' for gzip and 'tgz' for tar.gz.
    :param expected_type: The target type to load the objects into, e.g. list
    :param header_comment_token: Token used for header comments to be skipped
    :param select_query: JSONPath query to select specific objects from the loaded data.
    :return: A list of dictionaries representing the loaded objects.
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)

    for url_scheme in ["http", "https", "ftp"]:
        if file_path.startswith(f"{url_scheme}://"):
            return load_objects_from_url(
                file_path,
                format=format,
                expected_type=expected_type,
            )

    if isinstance(format, str):
        format = Format(format)

    all_objects = []

    if compression == "tgz":
        with tarfile.open(file_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        content = io.TextIOWrapper(f)
                        member_format = Format.guess_format(member.name) if not format else format
                        logger.debug(f"Processing tar member {member.name} with format {member_format}")
                        all_objects.extend(process_file(content, member_format, expected_type, header_comment_token))
    else:
        if Path(file_path).is_dir():
            raise ValueError(f"{file_path} is a dir, which is invalid for {format}")
        open_func = gzip.open if compression == "gz" else open
        format = Format.guess_format(file_path) if not format else format
        mode = "rb" if (format and format.is_binary_format()) or compression == "gz" else "r"
        with open_func(file_path, mode) if file_path != "-" else sys.stdin as f:
            if compression == "gz" and mode == "r":
                f = io.TextIOWrapper(f)
            all_objects = process_file(f, format, expected_type, header_comment_token)

    logger.debug(f"Loaded {len(all_objects)} objects from {file_path}")
    if select_query:
        import jsonpath_ng as jp

        path_expr = jp.parse(select_query)
        new_objs = []
        for obj in all_objects:
            for match in path_expr.find(obj):
                logging.debug(f"Match: {match.value}")
                if isinstance(match.value, list):
                    new_objs.extend(match.value)
                else:
                    new_objs.append(match.value)
        all_objects = new_objs
    return all_objects


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
    data: Union[List[Dict[str, Any]], Dict[str, Any], pd.DataFrame, List[BaseModel]],
    format: Optional[Union[Format, str]] = Format.YAML,
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
        return data.to_string(max_rows=None)

    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient="records")

    if isinstance(data, BaseModel):
        data = data.model_dump()

    if data and isinstance(data, list) and isinstance(data[0], BaseModel):
        data = [d.model_dump() if isinstance(d, BaseModel) else d for d in data]

    if isinstance(data, dict) and format in [Format.TSV, Format.CSV]:
        data = [data]

    if isinstance(data, BaseModel):
        data = data.model_dump()

    if format == Format.JSON:
        return json.dumps(data, indent=2, default=str)
    elif format == Format.JSONL:
        return "\n".join(json.dumps(obj) for obj in data)
    elif format == Format.PYTHON:
        return str(data)
    elif format == Format.MARKDOWN:

        def as_markdown(obj: dict):
            return "## Object\n\n" + "\n".join([f" * {k}: {v}" for k, v in obj.items()])

        return "\n\n".join([as_markdown(obj) for obj in data]) if isinstance(data, list) else as_markdown(data)
    elif format == Format.TABLE:
        from tabulate import tabulate

        return tabulate(pd.DataFrame(data), headers="keys", tablefmt="psql")
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
    return Format.guess_format(path)


def split_document(doc: str, delimiter: str):
    """
    Split a document into parts based on a delimiter.

    :param doc: The document to split.
    :param delimiter: The delimiter.
    :return: The parts of the document.
    """
    return doc.split(delimiter)
