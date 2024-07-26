import logging
import sys
import warnings
from typing import Optional

import click
import yaml
from linkml_runtime.dumpers import json_dumper
from pydantic import BaseModel

from linkml_store import Client
from linkml_store.api import Collection, Database
from linkml_store.api.queries import Query
from linkml_store.index import get_indexer
from linkml_store.index.implementations.simple_indexer import SimpleIndexer
from linkml_store.index.indexer import Indexer
from linkml_store.utils.format_utils import Format, guess_format, load_objects, render_output, write_output
from linkml_store.utils.object_utils import object_path_update
from linkml_store.utils.pandas_utils import facet_summary_to_dataframe_unmelted

index_type_option = click.option(
    "--index-type",
    "-t",
    default="simple",
    show_default=True,
    help="Type of index to create. Values: simple, llm",
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
def cli(ctx, verbose: int, quiet: bool, stacktrace: bool, database, collection, config, set, **kwargs):
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
    client = Client().from_config(config, **kwargs) if config else Client()
    settings = ContextSettings(client=client, database_name=database, collection_name=collection)
    ctx.obj["settings"] = settings
    # DEPRECATED
    ctx.obj["client"] = client
    ctx.obj["database"] = database
    ctx.obj["collection"] = collection
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
        # settings.database = db
        # DEPRECATED
        ctx.obj["database_obj"] = db
        if collection:
            collection_obj = db.get_collection(collection)
            ctx.obj["collection_obj"] = collection_obj
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
@click.argument("files", type=click.Path(exists=True), nargs=-1)
@click.option("--format", "-f", type=format_choice, help="Input format")
@click.option("--object", "-i", multiple=True, help="Input object as YAML")
@click.pass_context
def insert(ctx, files, object, format):
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
    objects = []
    if not files and not object:
        files = ["-"]
    for file_path in files:
        if format:
            objects = load_objects(file_path, format=format)
        else:
            objects = load_objects(file_path)
        logger.info(f"Inserting {len(objects)} objects from {file_path} into collection '{collection.alias}'.")
        collection.insert(objects)
        click.echo(f"Inserted {len(objects)} objects from {file_path} into collection '{collection.alias}'.")
    if object:
        for object_str in object:
            logger.info(f"Parsing: {object_str}")
            objects = yaml.safe_load(object_str)
            collection.insert(objects)
            click.echo(f"Inserted {len(objects)} objects from {object_str} into collection '{collection.alias}'.")
    collection.commit()


@cli.command()
@click.argument("files", type=click.Path(exists=True), nargs=-1)
@click.option("--format", "-f", type=format_choice, help="Input format")
@click.option("--object", "-i", multiple=True, help="Input object as YAML")
@click.pass_context
def store(ctx, files, object, format):
    """Store objects from files (JSON, YAML, TSV) into the specified collection."""
    settings = ctx.obj["settings"]
    db = settings.database
    if not files and not object:
        files = ["-"]
    for file_path in files:
        if format:
            objects = load_objects(file_path, format=format)
        else:
            objects = load_objects(file_path)
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
@click.option("--limit", "-l", type=click.INT, help="Maximum number of results to return")
@click.option("--output-type", "-O", type=format_choice, default="json", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.pass_context
def query(ctx, where, limit, output_type, output):
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
    query = Query(from_table=collection.alias, where_clause=where_clause, limit=limit)
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
@click.option("--limit", "-l", type=click.INT, help="Maximum number of results to return")
@click.option("--output-type", "-O", type=format_choice, default="json", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--columns", "-S", help="Columns to facet on")
@click.option("--wide/--no-wide", "-U/--no-U", default=False, show_default=True, help="Wide table")
@click.pass_context
def fq(ctx, where, limit, columns, output_type, wide, output):
    """
    Query facets from the specified collection.

    :param ctx:
    :param where:
    :param limit:
    :param columns:
    :param output_type:
    :param output:
    :return:
    """
    collection = ctx.obj["settings"].collection
    where_clause = yaml.safe_load(where) if where else None
    columns = columns.split(",") if columns else None
    if columns:
        columns = [col.strip() for col in columns]
        columns = [(tuple(col.split("+")) if "+" in col else col) for col in columns]
    logger.info(f"Faceting on columns: {columns}")
    results = collection.query_facets(where_clause, facet_columns=columns, limit=limit)
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
@click.option("--limit", "-l", type=click.INT, help="Maximum number of search results")
@click.option("--output-type", "-O", type=format_choice, default="json", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--auto-index/--no-auto-index", default=False, show_default=True, help="Automatically index the collection"
)
@index_type_option
@click.pass_context
def search(ctx, search_term, where, limit, index_type, output_type, output, auto_index):
    """Search objects in the specified collection."""
    collection = ctx.obj["settings"].collection
    ix = get_indexer(index_type)
    logger.info(f"Attaching index to collection {collection.alias}: {ix.model_dump()}")
    collection.attach_indexer(ix, auto_index=auto_index)
    result = collection.search(search_term, where=where, limit=limit)
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
@click.pass_context
def validate(ctx, output_type, output):
    """Validate objects in the specified collection."""
    collection = ctx.obj["settings"].collection
    logger.info(f"Validating collection {collection.alias}")
    validation_results = [json_dumper.to_dict(x) for x in collection.iter_validate_collection()]
    output_data = render_output(validation_results, output_type)
    if output:
        with open(output, "w") as f:
            f.write(output_data)
        click.echo(f"Validation results saved to {output}")
    else:
        click.echo(output_data)


if __name__ == "__main__":
    cli()
