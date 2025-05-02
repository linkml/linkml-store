import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple

import click
import yaml
from linkml_runtime.dumpers import json_dumper
from linkml_runtime.utils.formatutils import underscore
from pydantic import BaseModel

from linkml_store import Client
from linkml_store.api import Collection, Database
from linkml_store.api.config import ClientConfig
from linkml_store.api.queries import Query
from linkml_store.index import get_indexer
from linkml_store.index.implementations.simple_indexer import SimpleIndexer
from linkml_store.index.indexer import Indexer
from linkml_store.inference import get_inference_engine
from linkml_store.inference.evaluation import evaluate_predictor, score_text_overlap
from linkml_store.inference.inference_config import InferenceConfig
from linkml_store.inference.inference_engine import ModelSerialization
from linkml_store.utils.format_utils import Format, guess_format, load_objects, render_output, write_output
from linkml_store.utils.object_utils import object_path_update
from linkml_store.utils.pandas_utils import facet_summary_to_dataframe_unmelted

DEFAULT_LOCAL_CONF_PATH = Path("linkml.yaml")
# global path is ~/.linkml.yaml in the user's home directory
DEFAULT_GLOBAL_CONF_PATH = Path("~/.linkml.yaml").expanduser()

index_type_option = click.option(
    "--index-type",
    "-t",
    default="simple",
    show_default=True,
    help="Type of index to create. Values: simple, llm",
)
json_select_query_option = click.option(
    "--json-select-query",
    "-J",
    help="JSON SELECT query",
)

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="duckdb_engine")


class ContextSettings(BaseModel):
    """
    Context object for CLI commands.
    """

    client: Client
    database_name: Optional[str] = None
    collection_name: Optional[str] = None

    @property
    def database(self) -> Optional[Database]:
        """
        Get the database object.
        :return:
        """
        name = self.database_name
        if name is None:
            # if len(self.client.databases) > 1:
            #    raise ValueError("Database must be specified if there are multiple databases.")
            if not self.client.databases:
                return None
            name = list(self.client.databases.keys())[0]
        return self.client.get_database(name)

    @property
    def collection(self) -> Optional[Collection]:
        """
        Get the collection object.
        :return:
        """
        name = self.collection_name
        if name is None:
            # if len(self.database.list_collections()) > 1:
            #    raise ValueError("Collection must be specified if there are multiple collections.")
            if not self.database:
                return None
            if not self.database.list_collections():
                return None
            name = list(self.database.list_collections())[0]
        return self.database.get_collection(name)

    class Config:
        arbitrary_types_allowed = True


# format_choice = click.Choice(["json", "yaml", "tsv"])
format_choice = click.Choice([f.value for f in Format])


include_internal_option = click.option("--include-internal/--no-include-internal", default=False, show_default=True)


@click.group()
@click.option("--database", "-d", help="Database name")
@click.option("--collection", "-c", help="Collection name")
@click.option("--input", "-i", help="Input file (alternative to database/collection)")
@click.option("--schema", "-S", help="Path to schema (LinkML yaml)")
@click.option("--config", "-C", type=click.Path(exists=True), help="Path to the configuration file")
@click.option("--set", help="Metadata settings in the form PATHEXPR=value", multiple=True)
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet/--no-quiet")
@click.option("--base-dir", "-B", help="Base directory for the client configuration")
@click.option(
    "--stacktrace/--no-stacktrace",
    default=False,
    show_default=True,
    help="If set then show full stacktrace on error",
)
@click.pass_context
def cli(ctx, verbose: int, quiet: bool, stacktrace: bool, database, collection, schema, config, set, input, **kwargs):
    """A CLI for interacting with the linkml-store."""
    if not stacktrace:
        sys.tracebacklimit = 0
    logger = logging.getLogger()
    # Set handler for the root logger to output to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    # Clear existing handlers to avoid duplicate messages if function runs multiple times
    logger.handlers = []

    # Add the newly created console handler to the logger
    logger.addHandler(console_handler)
    if verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    if quiet:
        logger.setLevel(logging.ERROR)
    ctx.ensure_object(dict)
    if input:
        database = "duckdb"  # default: store in duckdb
        if input.startswith("http"):
            parts = input.split("/")
            collection = parts[-1]
            collection = collection.split(".")[0]
        else:
            stem = underscore(Path(input).stem)
            collection = stem
        logger.info(f"Using input file: {input}, " f"default storage is {database} and collection is {collection}")
        config = ClientConfig(databases={"duckdb": {"collections": {stem: {"source": {"local_path": input}}}}})
    if config is None and DEFAULT_LOCAL_CONF_PATH.exists():
        config = DEFAULT_LOCAL_CONF_PATH
    if config is None and DEFAULT_GLOBAL_CONF_PATH.exists():
        config = DEFAULT_GLOBAL_CONF_PATH
    if config == ".":
        config = None
    if not collection and database and "::" in database:
        database, collection = database.split("::")

    client = Client().from_config(config, **kwargs) if config else Client()
    settings = ContextSettings(client=client, database_name=database, collection_name=collection)
    ctx.obj["settings"] = settings
    if schema:
        db = settings.database
        db.set_schema_view(schema)
    if settings.database_name:
        db = client.get_database(database)
        if set:
            for expr in set:
                if "=" not in expr:
                    raise ValueError(f"Expression must be of form PARAM=VALUE. Got: {expr}")
                path, val = expr.split("=", 1)
                val = yaml.safe_load(val)
                logger.info(f"Setting {path} to {val}")
                db.metadata = object_path_update(db.metadata, path, val)
    if not settings.database_name:
        # if len(client.databases) != 1:
        #    raise ValueError("Database must be specified if there are multiple databases.")
        if client.databases:
            settings.database_name = list(client.databases.keys())[0]
    if not settings.collection_name:
        # if len(settings.database.list_collections()) != 1:
        #    raise ValueError("Collection must be specified if there are multiple collections.")
        if settings.database and settings.database.list_collections():
            collection = settings.database.list_collections()[0]
            settings.collection_name = collection.alias


@cli.command()
@click.pass_context
def drop(ctx):
    """
    Drop database and all its collections.
    """
    database = ctx.obj["settings"].database
    database.drop()


@cli.command()
@click.argument("files", type=click.Path(), nargs=-1)
@click.option("--replace/--no-replace", default=False, show_default=True, help="Replace existing objects")
@click.option("--format", "-f", type=format_choice, help="Input format")
@click.option("--object", "-i", multiple=True, help="Input object as YAML")
@click.option("--source-field", help="If provided, inject file path source as this field")
@json_select_query_option
@click.pass_context
def insert(ctx, files, replace, object, format, source_field, json_select_query):
    """Insert objects from files (JSON, YAML, TSV) into the specified collection.

    Using a configuration:

        linkml-store -C config.yaml -c genes insert data/genes/*.json

    Note: if you don't provide a schema this will be inferred, but it is
    usually better to provide an explicit schema
    """
    settings = ctx.obj["settings"]
    collection = settings.collection
    if not collection:
        raise ValueError("Collection must be specified.")
    if not files and not object:
        files = ["-"]
    load_objects_args = {}
    if json_select_query:
        load_objects_args["select_query"] = json_select_query
    for file_path in files:
        if format:
            objects = load_objects(file_path, format=format, **load_objects_args)
        else:
            objects = load_objects(file_path, **load_objects_args)
        if source_field:
            for obj in objects:
                obj[source_field] = str(file_path)
        logger.info(f"Inserting {len(objects)} objects from {file_path} into collection '{collection.alias}'.")
        if replace:
            collection.replace(objects)
        else:
            collection.insert(objects)
        click.echo(f"Inserted {len(objects)} objects from {file_path} into collection '{collection.alias}'.")
    if object:
        for object_str in object:
            logger.info(f"Parsing: {object_str}")
            objects = yaml.safe_load(object_str)
            if not isinstance(objects, list):
                objects = [objects]
            if replace:
                collection.replace(objects)
            else:
                collection.insert(objects)
            click.echo(f"Inserted {len(objects)} objects from {object_str} into collection '{collection.alias}'.")
    collection.commit()


@cli.command()
@click.argument("files", type=click.Path(exists=True), nargs=-1)
@click.option("--format", "-f", type=format_choice, help="Input format")
@click.option("--object", "-i", multiple=True, help="Input object as YAML")
@json_select_query_option
@click.pass_context
def store(ctx, files, object, format, json_select_query):
    """Store objects from files (JSON, YAML, TSV) into the database.

    Note: this is similar to insert, but a collection does not need to be specified.

    For example, assume that `my-collection` is a dict with multiple keys,
    and we want one collection per key:

        linkml-store -d my.ddb store my-collection.yaml

    Loading JSON (e.g OBO-JSON), with a --json-select-query:

        linkml-store -d cl.ddb  store -J graphs  cl.obo.json

    Loading XML (e.g OWL-XML), with a --json-select-query:

        linkml-store -d cl.ddb  store -J Ontology  cl.owx

    Because the XML uses a top level Ontology, with multiple

    """
    settings = ctx.obj["settings"]
    db = settings.database
    if not files and not object:
        files = ["-"]
    load_objects_args = {}
    if json_select_query:
        load_objects_args["select_query"] = json_select_query
    for file_path in files:
        if format:
            objects = load_objects(file_path, format=format, **load_objects_args)
        else:
            objects = load_objects(file_path, **load_objects_args)
        logger.info(f"Inserting {len(objects)} objects from {file_path} into database '{db}'.")
        for obj in objects:
            db.store(obj)
        click.echo(f"Inserted {len(objects)} objects from {file_path} into database '{db}'.")
    if object:
        for object_str in object:
            logger.info(f"Parsing: {object_str}")
            objects = yaml.safe_load(object_str)
            for obj in objects:
                db.store(obj)
            click.echo(f"Inserted {len(objects)} objects from {object_str} into collection '{db.name}'.")


@cli.command(name="import")
@click.option("--format", "-f", help="Input format")
@click.pass_context
@click.argument("files", type=click.Path(exists=True), nargs=-1)
def import_database(ctx, files, format):
    """Imports a database from a dump.

    See the `export` command for a full list of supported formats. The same
    formats are generally supported for imports.
    """
    settings = ctx.obj["settings"]
    db = settings.database
    if not files and not object:
        files = ["-"]
    for file_path in files:
        db.import_database(file_path, source_format=format)


@cli.command()
@click.option("--output-type", "-O", type=format_choice, default="json", help="Output format")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output file path")
@click.pass_context
def export(ctx, output_type, output):
    """Exports a database to a standard dump format.

    Example:

        linkml-store -d duckdb:///countries.db export -O yaml -o countries.yaml

    Export format will be guessed from extension if not specified

    Example:

        linkml-store -d duckdb:///countries.db export -o countries.json

    Tree formats such as YAML and JSON can natively store an entire database; each collection
    will be a distinct key in the database.

    Additionally, native dump formats can be used:

    Example:

        linkml-store -d duckdb:///countries.db export -o countries -O duckdb

    Here, `countries` is a directory. This is equivalent to running EXPORT DATABASE
    (see https://duckdb.org/docs/sql/statements/export.html)
    """
    settings = ctx.obj["settings"]
    db = settings.database
    if output_type is None:
        output_type = guess_format(output)
    if output_type is None:
        raise ValueError(f"Output format must be specified can't be inferred from {output}.")
    db.export_database(output, target_format=output_type)


@cli.command()
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--output-type", "-O", type=format_choice, default="json", help="Output format")
@click.option("--other-database", "-D", required=False, help="Path to the other database")
@click.option("--other-collection", "-X", required=True, help="Name of the other collection")
@click.option("--identifier-attribute", "-I", required=False, help="Primary key name")
@click.pass_context
def diff(ctx, output, output_type, other_database, other_collection, identifier_attribute):
    """Diffs two collectoons to create a patch."""
    settings = ctx.obj["settings"]
    db = settings.database
    collection = settings.collection
    if not collection:
        raise ValueError("Collection must be specified.")
    other_db = settings.client.get_database(other_database) if other_database else db
    other_collection = other_db.get_collection(other_collection)
    if identifier_attribute:
        collection.set_identifier_attribute_name(identifier_attribute)
        other_collection.set_identifier_attribute_name(identifier_attribute)
    diff = collection.diff(other_collection)
    write_output(diff, output_type, target=output)


@cli.command()
@click.option("--identifier-attribute", "-I", required=False, help="Primary key name")
@click.argument("patch_files", type=click.Path(exists=True), nargs=-1)
@click.pass_context
def apply(ctx, patch_files, identifier_attribute):
    """
    Apply a patch to a collection.
    """
    settings = ctx.obj["settings"]
    collection = settings.collection
    if not collection:
        raise ValueError("Collection must be specified.")
    if identifier_attribute:
        collection.set_identifier_attribute_name(identifier_attribute)
    for patch_file in patch_files:
        patch_objs = load_objects(patch_file, expected_type=list)
        collection.apply_patches(patch_objs)


@cli.command()
@click.option("--where", "-w", type=click.STRING, help="WHERE clause for the query, as YAML")
@click.option("--select", "-s", type=click.STRING, help="SELECT clause for the query, as YAML")
@click.option("--limit", "-l", type=click.INT, help="Maximum number of results to return")
@click.option("--output-type", "-O", type=format_choice, default="json", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.pass_context
def query(ctx, where, select, limit, output_type, output):
    """Query objects from the specified collection.


    Leave the query field blank to return all objects in the collection.

    Examples:

        linkml-store -d duckdb:///countries.db -c countries query

    Queries can be specified in YAML, as basic key-value pairs

    Examples:

        linkml-store -d duckdb:///countries.db -c countries query -w 'code: NZ'

    More complex queries can be specified using MongoDB-style query syntax

    Examples:

        linkml-store -d file:. -c persons query  -w 'occupation: {$ne: Architect}'

    Finds all people who are not architects.
    """
    collection = ctx.obj["settings"].collection
    where_clause = yaml.safe_load(where) if where else None
    select_clause = yaml.safe_load(select) if select else None
    if select_clause:
        if isinstance(select_clause, str):
            select_clause = [select_clause]
        if not isinstance(select_clause, list):
            raise ValueError(f"SELECT clause must be a list. Got: {select_clause}")
    query = Query(from_table=collection.alias, select_cols=select_clause, where_clause=where_clause, limit=limit)
    result = collection.query(query)
    output_data = render_output(result.rows, output_type)
    if output:
        with open(output, "w") as f:
            f.write(output_data)
        click.echo(f"Query results saved to {output}")
    else:
        click.echo(output_data)


@cli.command()
@click.pass_context
@include_internal_option
def list_collections(ctx, **kwargs):
    db = ctx.obj["settings"].database
    for collection in db.list_collections(**kwargs):
        click.echo(collection.alias)
        click.echo(render_output(collection.metadata))


@cli.command()
@click.option("--where", "-w", type=click.STRING, help="WHERE clause for the query")
@click.option("--limit", "-l", type=click.INT, help="Maximum number of results to return per facet")
@click.option("--facet-min-count", "-M", type=click.INT, help="Minimum count for a facet to be included")
@click.option("--output-type", "-O", type=format_choice, default="json", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--columns", "-S", help="Columns to facet on. Comma-separated, join combined facets with +")
@click.option("--wide/--no-wide", "-U/--no-U", default=False, show_default=True, help="Wide table")
@click.pass_context
def fq(ctx, where, limit, columns, output_type, wide, output, **kwargs):
    """
    Query facet counts from the specified collection.

    Assuming your .linkml.yaml includes an entry mapping `phenopackets` to a
    mongodb

    Facet counts (all columns)

        linkml-store -d phenopackets fq

    Nested columns:

        linkml-store -d phenopackets fq subject.timeAtLastEncounter.age

    Compound keys:

        linkml-store -d phenopackets fq subject.sex+subject.timeAtLastEncounter.age

    """
    collection = ctx.obj["settings"].collection
    where_clause = yaml.safe_load(where) if where else None
    columns = columns.split(",") if columns else None
    if columns:
        columns = [col.strip() for col in columns]
        columns = [(tuple(col.split("+")) if "+" in col else col) for col in columns]
    logger.info(f"Faceting on columns: {columns}")
    results = collection.query_facets(where_clause, facet_columns=columns, facet_limit=limit, **kwargs)
    logger.info(f"Facet results: {results}")

    def _untuple(key):
        if isinstance(key, tuple):
            return "+".join([str(x) for x in key])
        return key

    if wide:
        results_obj = facet_summary_to_dataframe_unmelted(results)
    else:
        if output_type == Format.PYTHON.value:
            results_obj = results
        elif output_type in [Format.TSV.value, Format.CSV.value]:
            results_obj = []
            for fc, data in results.items():
                for v, c in data:
                    results_obj.append({"facet": fc, "value": v, "count": c})
        else:
            results_obj = {}
            for key, value in results.items():
                value_as_dict = {_untuple(v[0:-1]): v[-1] for v in value}
                results_obj[_untuple(key)] = value_as_dict
    output_data = render_output(results_obj, output_type)
    if output:
        with open(output, "w") as f:
            f.write(output_data)
        click.echo(f"Query results saved to {output}")
    else:
        click.echo(output_data)


@cli.command()
@click.option("--where", "-w", type=click.STRING, help="WHERE clause for the query")
@click.option("--limit", "-l", type=click.INT, help="Maximum number of results to return per facet")
@click.option("--output-type", "-O", type=format_choice, default="json", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--columns", "-S", help="Columns to facet on. Comma-separated, join combined facets with +")
@click.pass_context
def groupby(ctx, where, limit, columns, output_type, output, **kwargs):
    """
    Group by columns in the specified collection.

    Assume a simple triple model:

        linkml-store -d cl.ddb -c triple insert cl.owl

    This makes a flat subject/predicate/object table

    This can be grouped, e.g by subject:

        linkml-store -d cl.ddb -c triple groupby -s subject

    Or subject and predicate:

        linkml-store -d cl.ddb -c triple groupby -s '[subject,predicate]'

    """
    collection = ctx.obj["settings"].collection
    where_clause = yaml.safe_load(where) if where else None
    columns = columns.split(",") if columns else None
    if columns:
        columns = [col.strip() for col in columns]
        columns = [(tuple(col.split("+")) if "+" in col else col) for col in columns]
    logger.info(f"Group by: {columns}")
    result = collection.group_by(
        group_by_fields=columns,
        where_clause=where_clause,
        agg_map={},
        limit=limit,
        **kwargs,
    )
    logger.info(f"Group by results: {result}")
    output_data = render_output(result.rows, output_type)
    if output:
        with open(output, "w") as f:
            f.write(output_data)
        click.echo(f"Query results saved to {output}")
    else:
        click.echo(output_data)


def _get_index(index_type=None, **kwargs) -> Indexer:
    if index_type is None or index_type == "simple":
        return SimpleIndexer(name="test", **kwargs)
    else:
        raise ValueError(f"Unknown index type: {index_type}")


@cli.command()
@click.option("--where", "-w", type=click.STRING, help="WHERE clause for the query")
@click.option("--output-type", "-O", type=format_choice, default=Format.FORMATTED.value, help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--limit", "-l", default=-1, show_default=True, type=click.INT, help="Maximum number of results to return"
)
@click.pass_context
def describe(ctx, where, output_type, output, limit):
    """
    Describe the collection schema.
    """
    where_clause = yaml.safe_load(where) if where else None
    collection = ctx.obj["settings"].collection
    df = collection.find(where_clause, limit=limit).rows_dataframe
    write_output(df.describe(include="all").transpose(), output_type, target=output)


@cli.command()
@click.option("--where", "-w", type=click.STRING, help="WHERE clause for the query")
@click.option("--limit", "-l", type=click.INT, help="Maximum number of results to return")
@click.option("--output-type", "-O", type=format_choice, default="json", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--index", "-I", help="Attributes to index on in pivot")
@click.option("--columns", "-A", help="Attributes to use as columns in pivot")
@click.option("--values", "-V", help="Attributes to use as values in pivot")
@click.pass_context
def pivot(ctx, where, limit, index, columns, values, output_type, output):
    collection = ctx.obj["settings"].collection
    where_clause = yaml.safe_load(where) if where else None
    column_atts = columns.split(",") if columns else None
    value_atts = values.split(",") if values else None
    index_atts = index.split(",") if index else None
    results = collection.find(where_clause, limit=limit)
    pivoted = defaultdict(dict)
    for row in results.rows:
        index_key = tuple([row.get(att) for att in index_atts])
        column_key = tuple([row.get(att) for att in column_atts])
        value_key = tuple([row.get(att) for att in value_atts])
        pivoted[index_key][column_key] = value_key
    pivoted_objs = []

    def detuple(t: Tuple) -> Any:
        if len(t) == 1:
            return t[0]
        return str(t)

    for index_key, data in pivoted.items():
        obj = {att: key for att, key in zip(index_atts, index_key)}
        for column_key, value_key in data.items():
            obj[detuple(column_key)] = detuple(value_key)
        pivoted_objs.append(obj)
    write_output(pivoted_objs, output_type, target=output)


@cli.command()
@click.option("--where", "-w", type=click.STRING, help="WHERE clause for the query")
@click.option("--limit", "-l", type=click.INT, help="Maximum number of results to return")
@click.option("--output-type", "-O", type=format_choice, default="json", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--sample-field", "-I", help="Field to use as the sample identifier")
@click.option("--classification-field", "-L", help="Field to use as for classification")
@click.option(
    "--p-value-threshold",
    "-P",
    type=click.FLOAT,
    default=0.05,
    show_default=True,
    help="P-value threshold for enrichment",
)
@click.option(
    "--multiple-testing-correction",
    "-M",
    type=click.STRING,
    default="bh",
    show_default=True,
    help="Multiple test correction method",
)
@click.argument("samples", type=click.STRING, nargs=-1)
@click.pass_context
def enrichment(ctx, where, limit, output_type, output, sample_field, classification_field, samples, **kwargs):
    from linkml_store.utils.enrichment_analyzer import EnrichmentAnalyzer

    collection = ctx.obj["settings"].collection
    where_clause = yaml.safe_load(where) if where else None
    column_atts = [sample_field, classification_field]
    results = collection.find(where_clause, select_cols=column_atts, limit=-1)
    df = results.rows_dataframe
    ea = EnrichmentAnalyzer(df, sample_key=sample_field, classification_key=classification_field)
    if not samples:
        samples = df[sample_field].unique()
    enrichment_results = []
    for sample in samples:
        enriched = ea.find_enriched_categories(sample, **kwargs)
        for e in enriched:
            obj = {"sample": sample, **e.model_dump()}
            enrichment_results.append(obj)
    output_data = render_output(enrichment_results, output_type)
    if output:
        with open(output, "w") as f:
            f.write(output_data)
        click.echo(f"Search results saved to {output}")
    else:
        click.echo(output_data)


@cli.command()
@click.option("--output-type", "-O", type=format_choice, default=Format.YAML.value, help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--target-attribute", "-T", type=click.STRING, multiple=True, help="Target attributes for inference")
@click.option(
    "--feature-attributes", "-F", type=click.STRING, help="Feature attributes for inference (comma separated)"
)
@click.option("--training-collection", type=click.STRING, help="Collection to use for training")
@click.option("--inference-config-file", "-Y", type=click.Path(), help="Path to inference configuration file")
@click.option("--export-model", "-E", type=click.Path(), help="Export model to file")
@click.option("--load-model", "-L", type=click.Path(), help="Load model from file")
@click.option("--model-format", "-M", type=click.Choice([x.value for x in ModelSerialization]), help="Format for model")
@click.option("--training-test-data-split", "-S", type=click.Tuple([float, float]), help="Training/test data split")
@click.option(
    "--predictor-type", "-t", default="sklearn", show_default=True, type=click.STRING, help="Type of predictor"
)
@click.option("--evaluation-count", "-n", type=click.INT, help="Number of examples to evaluate over")
@click.option("--evaluation-match-function", help="Name of function to use for matching objects in eval")
@click.option("--query", "-q", type=click.STRING, help="query term")
@click.option("--where", "-w", type=click.STRING, help="query term")
@click.pass_context
def infer(
    ctx,
    inference_config_file,
    where,
    query,
    evaluation_count,
    evaluation_match_function,
    training_test_data_split,
    training_collection,
    predictor_type,
    target_attribute,
    feature_attributes,
    output_type,
    output,
    model_format,
    export_model,
    load_model,
):
    """
    Predict a complete object from a partial object.

    Currently two main prediction methods are provided: RAG and sklearn

    ## RAG:

    The RAG approach will use Retrieval Augmented Generation to inference the missing attributes of an object.

    Example:

        linkml-store  -i countries.jsonl inference -t rag  -q 'name: Uruguay'

    Result:

        capital: Montevideo, code: UY, continent: South America, languages: [Spanish]

    You can pass in configurations as follows:

        linkml-store  -i countries.jsonl inference -t rag:llm_config.model_name=llama-3  -q 'name: Uruguay'

    ## SKLearn:

    This uses scikit-learn (defaulting to simple decision trees) to do the prediction.

        linkml-store -i tests/input/iris.csv inference -t sklearn \
           -q '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
    """
    where_clause = yaml.safe_load(where) if where else None
    if query:
        query_obj = yaml.safe_load(query)
    else:
        query_obj = None
    collection = ctx.obj["settings"].collection
    if collection:
        atts = collection.class_definition().attributes.keys()
    else:
        atts = []
    if feature_attributes:
        features = feature_attributes.split(",")
        features = [f.strip() for f in features]
    else:
        if query_obj:
            features = query_obj.keys()
        else:
            features = None
    if target_attribute:
        target_attributes = list(target_attribute)
    else:
        target_attributes = [att for att in atts if att not in features]
    if model_format:
        model_format = ModelSerialization(model_format)
    if load_model:
        logger.info(f"Loading predictor from {load_model}")
        predictor = get_inference_engine(predictor_type)
        predictor = type(predictor).load_model(load_model)
    else:
        if inference_config_file:
            config = InferenceConfig.from_file(inference_config_file)
        else:
            config = InferenceConfig(target_attributes=target_attributes, feature_attributes=features)
        if training_test_data_split:
            config.train_test_split = training_test_data_split
        predictor = get_inference_engine(predictor_type, config=config)
        training_collection_obj = collection
        if training_collection:
            training_collection_obj = ctx.obj["settings"].database.get_collection(training_collection)
        if training_collection_obj:
            logger.info(f"Using collection: {training_collection_obj.alias} for inference")
            split = training_test_data_split or (1.0, 0.0)
            predictor.load_and_split_data(training_collection_obj, split=split)
        predictor.initialize_model()
    if export_model:
        logger.info(f"Exporting model to {export_model} in {model_format}")
        predictor.export_model(export_model, model_format)
    if not query_obj and where_clause is None:
        if not export_model and not evaluation_count:
            raise ValueError("Query or evaluate must be specified if not exporting model")
    if evaluation_count:
        if evaluation_match_function == "score_text_overlap":
            match_function_fn = score_text_overlap
        elif evaluation_match_function is not None:
            raise ValueError(f"Unknown match function: {evaluation_match_function}")
        else:
            match_function_fn = None
        outcome = evaluate_predictor(
            predictor, target_attributes, evaluation_count=evaluation_count, match_function=match_function_fn
        )
        print(f"Outcome: {outcome} // accuracy: {outcome.accuracy}")
    if query_obj:
        result = predictor.derive(query_obj)
        dumped_obj = result.model_dump(exclude_none=True)
        write_output([dumped_obj], output_type, target=output)
    if where_clause is not None:
        predicted_objs = []
        for query_obj in collection.find(where_clause).rows:
            result = predictor.derive(query_obj)
            predicted_objs.append(result.predicted_object)
        write_output(predicted_objs, output_type, target=output)


@cli.command()
@index_type_option
@click.option("--cached-embeddings-database", "-E", help="Path to the database where embeddings are cached")
@click.option("--text-template", "-T", help="Template for text embeddings")
@click.pass_context
def index(ctx, index_type, **kwargs):
    """
    Create an index over a collection.

    By default a simple trigram index is used.
    """
    collection = ctx.obj["settings"].collection
    ix = get_indexer(index_type, **kwargs)
    collection.attach_indexer(ix)


@cli.command()
@click.pass_context
@click.option("--output-type", "-O", type=format_choice, default="yaml", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def schema(ctx, output_type, output):
    """
    Show the schema for a database

    :param ctx:
    :param index_type:
    :return:
    """
    db = ctx.obj["settings"].database
    schema_dict = json_dumper.to_dict(db.schema_view.schema)
    output_data = render_output(schema_dict, output_type)
    if output:
        with open(output, "w") as f:
            f.write(output_data)
        click.echo(f"Schema saved to {output}")
    else:
        click.echo(output_data)


@cli.command()
@click.argument("search_term")
@click.option("--where", "-w", type=click.STRING, help="WHERE clause for the search")
@click.option("--select", "-s", type=click.STRING, help="SELECT clause for the query, as YAML")
@click.option("--limit", "-l", type=click.INT, help="Maximum number of search results")
@click.option("--output-type", "-O", type=format_choice, default="json", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--auto-index/--no-auto-index", default=False, show_default=True, help="Automatically index the collection"
)
@index_type_option
@click.pass_context
def search(ctx, search_term, where, select, limit, index_type, output_type, output, auto_index):
    """Search objects in the specified collection."""
    collection = ctx.obj["settings"].collection
    ix = get_indexer(index_type)
    logger.info(f"Attaching index to collection {collection.alias}: {ix.model_dump()}")
    collection.attach_indexer(ix, auto_index=auto_index)
    select_cols = yaml.safe_load(select) if select else None
    result = collection.search(search_term, where=where, select_cols=select_cols, limit=limit)
    output_data = render_output([{"score": row[0], **row[1]} for row in result.ranked_rows], output_type)
    if output:
        with open(output, "w") as f:
            f.write(output_data)
        click.echo(f"Search results saved to {output}")
    else:
        click.echo(output_data)


@cli.command()
@click.pass_context
def indexes(ctx):
    """
    Show the indexes for a collection.
    """
    collection = ctx.obj["settings"].collection
    for name, ix in collection.indexers.items():
        click.echo(f"{name}: {type(ix)}\n{ix.model_json()}")


@cli.command()
@click.option("--output-type", "-O", type=format_choice, default="json", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--collection-only/--no-collection-only",
    default=False,
    show_default=True,
    help="Only validate specified collection",
)
@click.option(
    "--ensure-referential-integrity/--no-ensure-referential-integrity",
    default=True,
    show_default=True,
    help="Ensure referential integrity",
)
@click.pass_context
def validate(ctx, output_type, output, collection_only, **kwargs):
    """Validate objects in the specified collection."""
    if collection_only:
        collection = ctx.obj["settings"].collection
        logger.info(f"Validating collection {collection.alias}")
        validation_results = [json_dumper.to_dict(x) for x in collection.iter_validate_collection(**kwargs)]
    else:
        db = ctx.obj["settings"].database
        validation_results = [json_dumper.to_dict(x) for x in db.validate_database(**kwargs)]
    output_data = render_output(validation_results, output_type)
    if output:
        with open(output, "w") as f:
            f.write(output_data)
        click.echo(f"Validation results saved to {output}")
    else:
        click.echo(output_data)


if __name__ == "__main__":
    cli()
