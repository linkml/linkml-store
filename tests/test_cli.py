import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from linkml_store import Client
from linkml_store.cli import cli
from tests import INPUT_DIR, OUTPUT_DIR
from tests.conftest import JSON_FILE


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def config_file():
    return os.path.join(INPUT_DIR, "config.yaml")


@pytest.fixture
def test_files():
    return [
        JSON_FILE,
    ]


@pytest.fixture
def output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def test_help_option(cli_runner):
    """Ensures --help works."""
    result = cli_runner.invoke(cli, ["--help"])
    assert result.exit_code == 0


def test_insert_and_query_command(cli_runner, config_file, test_files, output_dir):
    """Ensures insert and query commands work together."""
    add_dupes = False
    output_file = os.path.join(output_dir, "query_results.json")
    # db = "duckdb:///:memory:"
    db_path = f"{output_dir}/test.db"
    db_name = f"duckdb:///{db_path}"
    collection_name = "test_collection"
    Path(db_path).unlink(missing_ok=True)
    # Insert objects
    result = cli_runner.invoke(
        cli,
        [
            "--database",
            db_name,
            "--collection",
            collection_name,
            "insert",
            *test_files,
        ],
    )
    if result.exit_code != 0:
        print(f"ERR: {result.stderr}")
        print(f"OUT: {result.stdout}")
        print(f"ALL: {result.output}")
    assert result.exit_code == 0

    if add_dupes:
        # dupes
        result = cli_runner.invoke(
            cli,
            [
                "--database",
                db_name,
                "--collection",
                collection_name,
                "insert",
                *test_files,
            ],
        )
        assert result.exit_code == 0

    # Query objects
    result = cli_runner.invoke(
        cli,
        [
            "--database",
            db_name,
            "--collection",
            collection_name,
            "query",
            "--where",
            "name: John",
            "--limit",
            "10",
            "--output-type",
            "json",
            "--output",
            output_file,
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(output_file)

    # Check query results
    # client = Client().from_config(config_file)
    if db_name != "duckdb:///:memory:":
        client = Client()
        # print(f"Getting db {db_name}")
        database = client.get_database(db_name)
        collection = database.get_collection(collection_name)
        objects = collection.find({"name": "John"}).rows
        assert len(objects) == 1  # Assuming only one object matches the query

    output_file = os.path.join(output_dir, "search_results.yaml")

    result = cli_runner.invoke(
        cli,
        [
            "--database",
            db_name,
            "--collection",
            collection_name,
            "search",
            "--auto-index",
            "John",
            # "--where",
            #'{"occupation": "Engineer"}',
            "--limit",
            "5",
            "--output-type",
            "yaml",
            "--output",
            output_file,
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(output_file)


def test_store(cli_runner, output_dir):
    """Ensures store command works."""
    db_path = f"{output_dir}/test.db"
    Path(db_path).unlink(missing_ok=True)
    database_handle = f"duckdb:///{db_path}"
    input_path = INPUT_DIR / "nested.yaml"
    result = cli_runner.invoke(
        cli,
        [
            "-d",
            database_handle,
            "store",
            str(input_path),
        ],
    )
    assert result.exit_code == 0
    schema_output_path = os.path.join(output_dir, "schema_output.yaml")
    result = cli_runner.invoke(
        cli,
        [
            "-d",
            database_handle,
            "schema",
            "-o",
            schema_output_path,
        ],
    )
    assert result.exit_code == 0
    schema_dict = yaml.safe_load(Path(schema_output_path).read_text())
    classes = schema_dict["classes"]
    assert len(classes) == 3
    assert set(classes.keys()) == {"about", "persons", "organizations"}


def test_store_explicit_schema(cli_runner, output_dir):
    """Ensures store command works, using explicit schema."""
    db_path = f"{output_dir}/test.store.db"
    Path(db_path).unlink(missing_ok=True)
    database_handle = f"duckdb:///{db_path}"
    input_path = INPUT_DIR / "nested.yaml"
    input_schema_path = INPUT_DIR / "nested.schema.yaml"
    # store the objects, using the schema
    result = cli_runner.invoke(
        cli,
        [
            "-d",
            database_handle,
            "--set",
            f"schema_location={input_schema_path}",
            "store",
            str(input_path),
        ],
    )
    assert result.exit_code == 0
    # now export the schema
    schema_output_path = os.path.join(output_dir, "schema_output.yaml")
    result = cli_runner.invoke(
        cli,
        [
            "-d",
            database_handle,
            "schema",
            "-o",
            schema_output_path,
        ],
    )
    assert result.exit_code == 0
    schema_dict = yaml.safe_load(Path(schema_output_path).read_text())
    classes = schema_dict["classes"]
    # note we have intentionally "lost" the original container
    assert len(classes) == 3
    assert set(classes.keys()) == {"about", "persons", "organizations"}


@pytest.mark.integration
def test_infer_using_rag(cli_runner, output_dir):
    input_path = INPUT_DIR / "countries" / "countries.jsonl"
    output_path = OUTPUT_DIR / "rag_inference_results.yaml"
    # store the objects, using the schema
    result = cli_runner.invoke(
        cli,
        [
            "-i",
            str(input_path),
            "infer",
            "-t",
            "rag",
            "-q",
            "name: Uruguay",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0
    result = yaml.safe_load(open(output_path))
    obj = result["predicted_object"]
    assert obj["capital"] == "Montevideo"


def test_query_with_sql_flag(cli_runner, output_dir):
    """Test raw SQL query execution via CLI."""
    db_path = f"{output_dir}/sql_test.db"
    Path(db_path).unlink(missing_ok=True)
    db_name = f"duckdb:///{db_path}"

    # First insert some data
    result = cli_runner.invoke(
        cli,
        ["-d", db_name, "-c", "persons", "insert", JSON_FILE],
    )
    assert result.exit_code == 0

    # Query using raw SQL
    output_file = f"{output_dir}/sql_results.json"
    result = cli_runner.invoke(
        cli,
        ["-d", db_name, "query", "--sql", "SELECT * FROM persons WHERE name = 'John'", "-o", output_file],
    )
    if result.exit_code != 0:
        print(f"Error: {result.output}")
    assert result.exit_code == 0
    assert os.path.exists(output_file)

    # Verify results
    import json
    with open(output_file) as f:
        results = json.load(f)
    assert len(results) == 1
    assert results[0]["name"] == "John"


def test_sql_flag_mutual_exclusivity(cli_runner, output_dir):
    """Test that --sql cannot be combined with --where."""
    db_path = f"{output_dir}/sql_mutex_test.db"
    Path(db_path).unlink(missing_ok=True)
    db_name = f"duckdb:///{db_path}"

    result = cli_runner.invoke(
        cli,
        ["-d", db_name, "query", "--sql", "SELECT 1", "--where", "x: 1"],
    )
    assert result.exit_code != 0
    assert "--sql cannot be combined" in result.output


def test_sql_requires_database(cli_runner):
    """Test that --sql requires a database to be specified."""
    result = cli_runner.invoke(
        cli,
        ["query", "--sql", "SELECT 1"],
    )
    assert result.exit_code != 0
    assert "Database must be specified" in result.output


def test_query_without_collection_requires_sql(cli_runner, output_dir):
    """Test that query without collection requires --sql."""
    db_path = f"{output_dir}/sql_nocoll_test.db"
    Path(db_path).unlink(missing_ok=True)
    db_name = f"duckdb:///{db_path}"

    result = cli_runner.invoke(
        cli,
        ["-d", db_name, "query", "--where", "x: 1"],
    )
    assert result.exit_code != 0
    assert "Collection must be specified" in result.output


@pytest.fixture
def countries_db(cli_runner, tmp_path):
    """Handle of a DuckDB database with the countries loaded into the collection `countries`."""
    from tests import COUNTRIES_DATA_JSONL

    handle = f"duckdb:///{tmp_path / 'countries.db'}"
    result = cli_runner.invoke(cli, ["-d", handle, "-c", "countries", "insert", str(COUNTRIES_DATA_JSONL)])
    assert result.exit_code == 0, result.output
    return handle


def test_groupby_where(cli_runner, countries_db):
    """groupby applies --where and --limit."""
    import json

    args = ["-d", countries_db, "-c", "countries", "groupby", "-S", "continent", "-O", "json"]
    result = cli_runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    groups = json.loads(result.output)
    assert len(groups) > 2

    result = cli_runner.invoke(cli, args + ["-w", "continent: Europe"])
    assert result.exit_code == 0, result.output
    filtered_groups = json.loads(result.output)
    assert [g["continent"] for g in filtered_groups] == ["Europe"]
    assert len(filtered_groups[0]["objects"]) > 1

    result = cli_runner.invoke(cli, args + ["-l", "2"])
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.output)) == 2


@pytest.mark.parametrize("columns", ["continent,code", "continent+code"])
def test_groupby_multiple_columns(cli_runner, countries_db, columns):
    import json

    args = ["-d", countries_db, "-c", "countries", "groupby", "-S", columns, "-w", "code: FR"]
    result = cli_runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    groups = json.loads(result.output)
    assert [(g["continent"], g["code"]) for g in groups] == [("Europe", "FR")]


@pytest.mark.parametrize("extra_args", [[], ["--collection-only"]])
def test_validate(cli_runner, countries_db, extra_args):
    pytest.importorskip("linkml.validator")
    result = cli_runner.invoke(cli, ["-d", countries_db, "-c", "countries", "validate", *extra_args])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.output) == []


@pytest.mark.parametrize(
    "file_name,extra_args,loader",
    [
        ("out.yaml", [], "yaml"),
        ("out.json", [], "json"),
        ("out.txt", ["-O", "yaml"], "yaml"),
    ],
)
def test_export_guesses_format(cli_runner, countries_db, tmp_path, file_name, extra_args, loader):
    """The export format comes from --output-type, or else from the file extension."""
    import json

    output = tmp_path / file_name
    result = cli_runner.invoke(cli, ["-d", countries_db, "export", "-o", str(output), *extra_args])
    assert result.exit_code == 0, result.output
    text = output.read_text()
    if loader == "json":
        assert text.startswith("{")
        data = json.loads(text)
    else:
        assert not text.startswith("{")
        data = yaml.safe_load(text)
    assert len(data["countries"]) == 20


def test_query_yaml_round_trip(cli_runner, countries_db, tmp_path):
    """Query results written as YAML can be inserted again."""
    output = tmp_path / "countries.yaml"
    args = ["-d", countries_db, "-c", "countries", "query", "-O", "yaml", "-o", str(output)]
    result = cli_runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    result = cli_runner.invoke(cli, ["-d", countries_db, "-c", "copies", "insert", str(output)])
    assert result.exit_code == 0, result.output
    assert "Inserted 20 objects" in result.output


@pytest.mark.parametrize("object_str", ["{persons: [{id: 1}, {id: 2}]}", "[{persons: [{id: 1}, {id: 2}]}]"])
def test_store_object_option(cli_runner, tmp_path, object_str):
    handle = f"duckdb:///{tmp_path / 'store.db'}"
    result = cli_runner.invoke(cli, ["-d", handle, "store", "--object", object_str])
    assert result.exit_code == 0, result.output
    result = cli_runner.invoke(cli, ["-d", handle, "-c", "persons", "query", "-O", "yaml"])
    assert result.exit_code == 0, result.output
    assert [row["id"] for row in yaml.safe_load_all(result.output)] == [1, 2]


@pytest.mark.parametrize(
    "args,message",
    [
        (["import"], "At least one file"),
        (["-c", "countries", "pivot"], "Missing option"),
        (["-c", "countries", "groupby"], "Missing option"),
    ],
)
def test_usage_errors(cli_runner, countries_db, args, message):
    result = cli_runner.invoke(cli, ["-d", countries_db, *args])
    assert result.exit_code == 2
    assert message in result.output


def test_schema_option_requires_database(cli_runner, tmp_path, monkeypatch):
    from tests import COUNTRIES_SCHEMA

    # no default configuration file in the working directory
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("linkml_store.cli.DEFAULT_GLOBAL_CONF_PATH", tmp_path / "no-such-config.yaml")
    result = cli_runner.invoke(cli, ["-S", str(COUNTRIES_SCHEMA), "schema"])
    assert result.exit_code == 2
    assert "--schema requires a database" in result.output


def test_pivot(cli_runner, countries_db):
    args = ["-d", countries_db, "-c", "countries", "pivot", "-I", "continent", "-A", "code", "-V", "name"]
    result = cli_runner.invoke(cli, [*args, "-w", "continent: Oceania", "-O", "yaml"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.output) == {"continent": "Oceania", "AU": "Australia", "NZ": "New Zealand"}


def test_describe_empty_collection(cli_runner, tmp_path):
    handle = f"duckdb:///{tmp_path / 'empty.db'}"
    result = cli_runner.invoke(cli, ["-d", handle, "-c", "nothing", "describe"])
    assert result.exit_code == 1
    assert "No objects found" in result.output


def test_search_sees_objects_inserted_after_indexing(cli_runner, tmp_path):
    """Each CLI call is a new process as far as the index is concerned; inserts must invalidate it."""
    base_args = ["-d", f"duckdb:///{tmp_path / 'search.db'}", "-c", "persons"]

    def run(*args):
        result = cli_runner.invoke(cli, [*base_args, *args])
        assert result.exit_code == 0, result.output
        return result.output

    run("insert", "--object", "[{id: 1, name: alpha scientist}, {id: 2, name: beta plumber}]")
    run("index", "-t", "simple")
    run("insert", "--object", "[{id: 3, name: gamma astronaut}]")
    hits = list(yaml.safe_load_all(run("search", "astronaut", "-t", "simple", "-O", "yaml")))
    assert hits[0]["id"] == 3
