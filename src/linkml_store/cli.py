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
from linkml_store.index.implementations.simple_indexer import SimpleIndexer
from linkml_store.index.indexer import Indexer
from linkml_store.utils.format_utils import Format, load_objects, render_output
from linkml_store.utils.object_utils import object_path_update

index_type_option = click.option("--index-type", "-t")

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


@click.group()
@click.option("--database", "-d", help="Database name")
@click.option("--collection", "-c", help="Collection name")
@click.option("--config", "-C", type=click.Path(exists=True), help="Path to the configuration file")
@click.option("--set", help="Metadata settings in the form PATHEXPR=value", multiple=True)
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet/--no-quiet")
@click.option(
    "--stacktrace/--no-stacktrace",
    default=False,
    show_default=True,
    help="If set then show full stacktrace on error",
)
@click.pass_context
def cli(ctx, verbose: int, quiet: bool, stacktrace: bool, database, collection, config, set):
    """A CLI for interacting with the linkml-store."""
    if not stacktrace:
        sys.tracebacklimit = 0
    logger = logging.getLogger()
    if verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    if quiet:
        logger.setLevel(logging.ERROR)
    ctx.ensure_object(dict)
    client = Client().from_config(config) if config else Client()
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
            settings.collection_name = collection.name


@cli.command()
@click.argument("files", type=click.Path(exists=True), nargs=-1)
@click.option("--format", "-f", type=format_choice, help="Input format")
@click.option("--object", "-i", multiple=True, help="Input object as YAML")
@click.pass_context
def insert(ctx, files, object, format):
    """Insert objects from files (JSON, YAML, TSV) into the specified collection."""
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
        logger.info(f"Inserting {len(objects)} objects from {file_path} into collection '{collection.name}'.")
        collection.insert(objects)
        click.echo(f"Inserted {len(objects)} objects from {file_path} into collection '{collection.name}'.")
    if object:
        for object_str in object:
            logger.info(f"Parsing: {object_str}")
            objects = yaml.safe_load(object_str)
            collection.insert(objects)
            click.echo(f"Inserted {len(objects)} objects from {object_str} into collection '{collection.name}'.")


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


@cli.command()
@click.option("--where", "-w", type=click.STRING, help="WHERE clause for the query")
@click.option("--limit", "-l", type=click.INT, help="Maximum number of results to return")
@click.option("--output-type", "-O", type=format_choice, default="json", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.pass_context
def query(ctx, where, limit, output_type, output):
    """Query objects from the specified collection."""
    collection = ctx.obj["settings"].collection
    where_clause = yaml.safe_load(where) if where else None
    query = Query(from_table=collection.name, where_clause=where_clause, limit=limit)
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
def list_collections(ctx):
    db = ctx.obj["settings"].database
    for collection in db.list_collections():
        click.echo(collection.name)
        click.echo(render_output(collection.metadata))


@cli.command()
@click.option("--where", "-w", type=click.STRING, help="WHERE clause for the query")
@click.option("--limit", "-l", type=click.INT, help="Maximum number of results to return")
@click.option("--output-type", "-O", type=format_choice, default="json", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--columns", "-S", help="Columns to facet on")
@click.pass_context
def fq(ctx, where, limit, columns, output_type, output):
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
            return "+".join(key)
        return key

    count_dict = {}
    for key, value in results.items():
        value_as_dict = {_untuple(v[0:-1]): v[-1] for v in value}
        count_dict[_untuple(key)] = value_as_dict
    output_data = render_output(count_dict, output_type)
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
@index_type_option
@click.pass_context
def index(ctx, index_type):
    """
    Create an index over a collection.

    :param ctx:
    :param index_type:
    :return:
    """
    collection = ctx.obj["settings"].collection
    ix = _get_index(index_type)
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
@index_type_option
@click.pass_context
def search(ctx, search_term, where, limit, index_type, output_type, output):
    """Search objects in the specified collection."""
    collection = ctx.obj["settings"].collection
    ix = _get_index(index_type)
    logger.info(f"Attaching index to collection {collection.name}: {ix.model_dump()}")
    collection.attach_indexer(ix, auto_index=False)
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
