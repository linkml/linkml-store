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
