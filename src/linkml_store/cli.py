import logging
import sys

import click
import yaml

from linkml_store import Client
from linkml_store.api.queries import Query
from linkml_store.index.implementations.simple_index import SimpleIndex
from linkml_store.index.index import Index
from linkml_store.utils.format_utils import load_objects, render_output

index_type_option = click.option('--index-type', '-t')


@click.group()
@click.option('--database', '-d', help='Database name')
@click.option('--collection', '-c', help='Collection name')
@click.option('--config', '-f', type=click.Path(exists=True), help='Path to the configuration file')
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet/--no-quiet")
@click.option(
    "--stacktrace/--no-stacktrace",
    default=False,
    show_default=True,
    help="If set then show full stacktrace on error",
)
@click.pass_context
def cli(ctx, verbose: int,
    quiet: bool,
    stacktrace: bool,
    database, collection, config
):
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
    ctx.obj['client'] = Client().from_config(config) if config else Client()
    ctx.obj['database'] = database
    ctx.obj['collection'] = collection


@cli.command()
@click.argument('files', type=click.Path(exists=True), nargs=-1)
@click.option('--format', '-f', type=click.Choice(['json', 'yaml', 'tsv']), help='Input format')
@click.pass_context
def insert(ctx, files, format):
    """Insert objects from files (JSON, YAML, TSV) into the specified collection."""
    db = ctx.obj['client'].get_database(ctx.obj['database'])
    collection = db.get_collection(ctx.obj['collection'])
    for file_path in files:
        if format:
            if format == 'json':
                objects = load_objects(file_path, format='json')
            elif format == 'yaml':
                objects = load_objects(file_path, format='yaml')
            elif format == 'tsv':
                objects = load_objects(file_path, format='tsv')
        else:
            objects = load_objects(file_path)
        collection.add(objects)
        click.echo(f"Inserted {len(objects)} objects from {file_path} into collection '{ctx.obj['collection']}'.")


@cli.command()
@click.option('--where', '-w', type=click.STRING, help='WHERE clause for the query')
@click.option('--limit', '-l', type=click.INT, help='Maximum number of results to return')
@click.option('--output-type', '-O', type=click.Choice(['json', 'yaml', 'tsv']), default='json', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def query(ctx, where, limit, output_type, output):
    """Query objects from the specified collection."""
    db = ctx.obj['client'].get_database(ctx.obj['database'])
    collection = db.get_collection(ctx.obj['collection'])
    where_clause = yaml.safe_load(where) if where else None
    query = Query(from_table=ctx.obj['collection'], where_clause=where_clause, limit=limit)
    result = collection.query(query)
    output_data = render_output(result.rows, output_type)
    if output:
        with open(output, 'w') as f:
            f.write(output_data)
        click.echo(f"Query results saved to {output}")
    else:
        click.echo(output_data)


@cli.command()
@click.option('--where', '-w', type=click.STRING, help='WHERE clause for the query')
@click.option('--limit', '-l', type=click.INT, help='Maximum number of results to return')
@click.option('--output-type', '-O', type=click.Choice(['json', 'yaml', 'tsv']), default='json', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--columns', '-S')
@click.pass_context
def fq(ctx, where, limit, columns, output_type, output):
    db = ctx.obj['client'].get_database(ctx.obj['database'])
    collection = db.get_collection(ctx.obj['collection'])
    where_clause = yaml.safe_load(where) if where else None
    count_dict = collection.query_facets(where_clause, facet_columns=columns.split(",") if columns else None, limit=limit)
    for key, value in count_dict.items():
        count_dict[key] = dict(value)
    output_data = render_output(count_dict, output_type)
    if output:
        with open(output, 'w') as f:
            f.write(output_data)
        click.echo(f"Query results saved to {output}")
    else:
        click.echo(output_data)

def _get_index(index_type=None, **kwargs) -> Index:
    if index_type is None or index_type == 'simple':
        return SimpleIndex(name="test", **kwargs)
    else:
        raise ValueError(f"Unknown index type: {index_type}")


@cli.command()
@index_type_option
@click.pass_context
def index(ctx, index_type):
    db = ctx.obj['client'].get_database(ctx.obj['database'])
    collection = db.get_collection(ctx.obj['collection'])
    ix = _get_index(index_type)
    collection.attach_index(ix)

@cli.command()
@click.argument('search_term')
@click.option('--where', '-w', type=click.STRING, help='WHERE clause for the search')
@click.option('--limit', '-l', type=click.INT, help='Maximum number of search results')
@click.option('--output-type', '-O', type=click.Choice(['json', 'yaml', 'tsv']), default='json', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@index_type_option
@click.pass_context
def search(ctx, search_term, where, limit, index_type, output_type, output):
    """Search objects in the specified collection."""
    db = ctx.obj['client'].get_database(ctx.obj['database'])
    collection = db.get_collection(ctx.obj['collection'])
    ix = _get_index(index_type)
    collection.attach_index(ix, auto_index=False)
    result = collection.search(search_term, where=where, limit=limit)
    output_data = render_output([row[1] for row in result.ranked_rows], output_type)
    if output:
        with open(output, 'w') as f:
            f.write(output_data)
        click.echo(f"Search results saved to {output}")
    else:
        click.echo(output_data)


if __name__ == '__main__':
    cli()