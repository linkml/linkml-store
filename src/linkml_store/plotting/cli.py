"""
Command-line interface for the plotting package.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import click

from linkml_store.plotting.heatmap import heatmap_from_file, export_heatmap_data
from linkml_store.utils.format_utils import Format

logger = logging.getLogger(__name__)


@click.group()
def plot_cli():
    """Plotting utilities for LinkML data."""
    pass


@plot_cli.command()
@click.argument("input_file", required=False)
@click.option("--x-column", "-x", required=True, help="Column to use for x-axis")
@click.option("--y-column", "-y", required=True, help="Column to use for y-axis")
@click.option("--value-column", "-v", help="Column containing values (if not provided, counts will be used)")
@click.option("--title", "-t", help="Title for the heatmap")
@click.option("--width", "-w", type=int, default=10, show_default=True, help="Width of the figure in inches")
@click.option("--height", "-h", type=int, default=8, show_default=True, help="Height of the figure in inches")
@click.option("--cmap", "-c", default="YlGnBu", show_default=True, help="Colormap to use")
@click.option("--output", "-o", required=True, help="Output file path")
@click.option("--format", "-f", help="Input file format")
@click.option("--dpi", type=int, default=300, show_default=True, help="DPI for output image")
@click.option("--square/--no-square", default=False, show_default=True, help="Make cells square")
@click.option("--annotate/--no-annotate", default=True, show_default=True, help="Annotate cells with values")
@click.option("--font-size", type=int, default=10, show_default=True, help="Font size for annotations and labels")
@click.option("--robust/--no-robust", default=False, show_default=True, help="Use robust quantiles for colormap scaling")
@click.option("--remove-duplicates/--no-remove-duplicates", default=True, show_default=True,
              help="Remove duplicate x,y combinations (default) or keep all occurrences")
@click.option("--cluster", type=click.Choice(["none", "both", "x", "y"]), default="none", show_default=True,
              help="Cluster axes: none (default), both, x-axis only, or y-axis only")
@click.option("--cluster-method", type=click.Choice(["complete", "average", "single", "ward"]), default="complete", show_default=True,
              help="Linkage method for hierarchical clustering")
@click.option("--cluster-metric", type=click.Choice(["euclidean", "correlation", "cosine", "cityblock"]), default="euclidean", show_default=True,
              help="Distance metric for clustering")
@click.option("--export-data", "-e", help="Export the heatmap data to this file")
@click.option("--export-format", "-E", type=click.Choice([f.value for f in Format]), default="csv", show_default=True,
              help="Format for exported data")
def heatmap(
    input_file: Optional[str],
    x_column: str,
    y_column: str,
    value_column: Optional[str],
    title: Optional[str],
    width: int,
    height: int,
    cmap: str,
    output: str,
    format: Optional[str],
    dpi: int,
    square: bool,
    annotate: bool,
    font_size: int,
    robust: bool,
    remove_duplicates: bool,
    cluster: str,
    cluster_method: str,
    cluster_metric: str,
    export_data: Optional[str],
    export_format: Union[str, Format],
):
    """
    Create a heatmap from a tabular data file.

    Examples:
      # From a file
      linkml-store plot heatmap data.csv -x species -y country -o heatmap.png

      # From stdin
      cat data.csv | linkml-store plot heatmap -x species -y country -o heatmap.png

    This will create a heatmap showing the frequency counts of species by country.
    If you want to use a specific value column instead of counts:

      linkml-store plot heatmap data.csv -x species -y country -v population -o heatmap.png
    """
    # Handle file path - if None, use stdin
    if input_file is None:
        input_file = "-"  # format_utils treats "-" as stdin
    
    # Convert 'none' to False for clustering parameter
    use_cluster = False if cluster == "none" else cluster
    
    # Create heatmap visualization
    fig, ax = heatmap_from_file(
        file_path=input_file,
        x_column=x_column,
        y_column=y_column,
        value_column=value_column,
        title=title,
        figsize=(width, height),
        cmap=cmap,
        output_file=output,
        format=format,
        dpi=dpi,
        square=square,
        annot=annotate,
        font_size=font_size,
        robust=robust,
        remove_duplicates=remove_duplicates,
        cluster=use_cluster,
        cluster_method=cluster_method,
        cluster_metric=cluster_metric,
    )
    
    # Export data if requested
    if export_data:
        # For export, reuse the data already loaded for the heatmap instead of loading again
        # This avoids the "I/O operation on closed file" error when input_file is stdin
        import pandas as pd
        from matplotlib.axes import Axes
        
        # Extract the data directly from the plot
        if hasattr(ax, 'get_figure') and hasattr(ax, 'get_children'):
            # Extract the heatmap data from the plot itself
            heatmap_data = {}
            for child in ax.get_children():
                if isinstance(child, plt.matplotlib.collections.QuadMesh):
                    # Get the colormap data
                    data_values = child.get_array()
                    rows = ax.get_yticks()
                    cols = ax.get_xticks()
                    row_labels = [item.get_text() for item in ax.get_yticklabels()]
                    col_labels = [item.get_text() for item in ax.get_xticklabels()]
                    
                    # Create a dataframe from the plot data
                    heatmap_df = pd.DataFrame(
                        index=[label for label in row_labels if label],
                        columns=[label for label in col_labels if label]
                    )
                    
                    # Fill in the values (if we can)
                    if len(data_values) == len(row_labels) * len(col_labels):
                        for i, row in enumerate(row_labels):
                            for j, col in enumerate(col_labels):
                                if row and col:  # Skip empty labels
                                    idx = i * len(col_labels) + j
                                    if idx < len(data_values):
                                        heatmap_df.at[row, col] = data_values[idx]
                    
                    # Reset index to make the y_column a regular column
                    result_df = heatmap_df.reset_index()
                    result_df.rename(columns={'index': y_column}, inplace=True)
                    
                    # Export the data
                    from linkml_store.utils.format_utils import write_output
                    records = result_df.to_dict(orient='records')
                    write_output(records, format=export_format, target=export_data)
                    click.echo(f"Heatmap data exported to {export_data}")
                    break
            else:
                # If we couldn't extract data from the plot, inform the user
                click.echo("Warning: Could not export data from the plot")
        else:
            click.echo("Warning: Could not export data from the plot")

    click.echo(f"Heatmap created at {output}")


if __name__ == "__main__":
    plot_cli()