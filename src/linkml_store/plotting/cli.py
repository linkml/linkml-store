"""
Command-line interface for the plotting package.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import click

from linkml_store.plotting.heatmap import heatmap_from_file, export_heatmap_data
from linkml_store.utils.format_utils import Format, load_objects
import pandas as pd

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
@click.option("--minimum-value", "-m", type=float, help="Minimum value to include in the heatmap")
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
@click.option("--remove-duplicates/--no-remove-duplicates", default=False, show_default=True,
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
    minimum_value: Optional[float],
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
        minimum_value=minimum_value,
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


@plot_cli.command()
@click.argument("input_file", required=False)
@click.option("--x-column", "-x", required=True, help="Column to use for x-axis")
@click.option("--bins", "-b", type=int, default=10, show_default=True, help="Number of bins for the histogram")
@click.option("--value-column", "-v", help="Column containing values (if not provided, counts will be used)")
@click.option("--x-log-scale/--no-x-log-scale", default=False, show_default=True, help="Use log scale for the x-axis")
@click.option("--y-log-scale/--no-y-log-scale", default=False, show_default=True, help="Use log scale for the y-axis")
@click.option("--title", "-t", help="Title for the heatmap")
@click.option("--width", "-w", type=int, default=10, show_default=True, help="Width of the figure in inches")
@click.option("--height", "-h", type=int, default=8, show_default=True, help="Height of the figure in inches")
@click.option("--output", "-o", required=True, help="Output file path")
def histogram(
    input_file: Optional[str],
    x_column: str,
    bins: int,
    value_column: Optional[str],
    x_log_scale: bool,
    y_log_scale: bool,
    title: Optional[str],
    width: int,
    height: int,
    output: str,
):
    """
    Create a histogram from a tabular data file.
    """
    # Handle file path - if None, use stdin
    if input_file is None:
        input_file = "-"  # format_utils treats "-" as stdin

    objs = load_objects(input_file)
    import pandas as pd
    df = pd.DataFrame(objs)

    # if the x column is a list, then translate it to the length of the list
    if isinstance(df[x_column].iloc[0], (list, tuple)):
        df[x_column] = df[x_column].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else x)

    # Debug: Check your DataFrame first
    print("DataFrame shape:", df.shape)
    print("DataFrame head:")
    print(df.head())
    print("\nColumn names:", df.columns.tolist())
    print("Data types:")
    print(df.dtypes)
    print("\nSize column info:")
    print("Unique values:", df[x_column].nunique())
    print("Sample values:", df[x_column].unique()[:10])
    print("Any null values?", df[x_column].isnull().sum())
    
    import matplotlib.pyplot as plt
    # Count the frequency of each size value
    size_counts = df[x_column].value_counts().sort_index()

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    if bins == 0:
        min_val = int(df[x_column].min())
        max_val = int(df[x_column].max())
        bin_edges = range(min_val, max_val + 2)  # +2 to include the last value
        plt.hist(df[x_column], bins=bin_edges, alpha=0.7, edgecolor='black', linewidth=0.5)
    else:
        plt.hist(df[x_column], bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.xlabel(x_column.replace('_', ' ').title())
    plt.ylabel('Frequency')
    plt.title(title or f'Distribution of {x_column}')

    if x_log_scale:
        plt.xscale('log')
    if y_log_scale:
        plt.yscale('log')
    
    # Add some stats to the plot
    mean_val = df[x_column].mean()
    median_val = df[x_column].median()
    plt.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.1f}')
    plt.axvline(median_val, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_val:.1f}')
    plt.legend()

    # Rotate x-axis labels if there are many unique sizes
    if len(size_counts) > 10:
        plt.xticks(rotation=45)
    plt.savefig(output, bbox_inches="tight", dpi=150)
    plt.close()


@plot_cli.command()
@click.argument("input_file", required=False)
@click.option("--x-column", "-x", required=True, help="Column to use for x-axis")
@click.option("--y-column", "-y", required=True, help="Column to use for y-axis")
@click.option("--x-log-scale/--no-x-log-scale", default=False, show_default=True, help="Use log scale for the x-axis")
@click.option("--y-log-scale/--no-y-log-scale", default=False, show_default=True, help="Use log scale for the y-axis")
@click.option("--value-column", "-v", help="Column containing values (if not provided, counts will be used)")
@click.option("--title", "-t", help="Title for the heatmap")
@click.option("--output", "-o", required=True, help="Output file path")
def boxplot_old(
    input_file: Optional[str],
    x_column: str,
    y_column: str,
    x_log_scale: bool,
    y_log_scale: bool,
    value_column: Optional[str],
    title: Optional[str],
    output: str,
):
    """
    Create a boxplot from a tabular data file.
    """
    # Handle file path - if None, use stdin
    if input_file is None:
        input_file = "-"  # format_utils treats "-" as stdin
        
    objs = load_objects(input_file)

    import pandas as pd
    df = pd.DataFrame(objs)

    # if y column is a list, explode it
    if isinstance(df[y_column].iloc[0], (list, tuple)):
        df[y_column] = df[y_column].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
        print("MADE A LIST INTO A SINGLE VALUE", df[y_column].head())
    if isinstance(df[x_column].iloc[0], (list, tuple)):
        df[x_column] = df[x_column].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
        print("MADE A LIST INTO A SINGLE VALUE", df[x_column].head())

    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df, x=x_column, y=y_column, 
                 # Outlier customization
                 flierprops={'marker': 'o',        # circle markers
                            'markerfacecolor': 'red',  # fill color
                            'markersize': 5,           # size
                            'alpha': 0.7})             # transparency

    if x_log_scale:
        plt.xscale('log')
    if y_log_scale:
        plt.yscale('log')

    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylabel(y_column.replace('_', ' ').title())
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight", dpi=150)
    plt.close()


@plot_cli.command()
@click.argument("input_file", required=False)
@click.option("--x-column", "-x", required=True, help="Column to use for x-axis")
@click.option("--y-column", "-y", required=False, help="Column to use for y-axis. If not specified, will count")
@click.option("--title", "-t", help="Title for the heatmap")
@click.option("--width", "-w", type=int, default=10, show_default=True, help="Width of the figure in inches")
@click.option("--height", "-h", type=int, default=8, show_default=True, help="Height of the figure in inches")
@click.option("--output", "-o", required=True, help="Output file path")
def barchart(input_file: Optional[str], x_column: str, y_column: str, title: Optional[str], width: int, height: int, output: str):
    """
    Create a barchart from a tabular data file.
    """
    # Handle file path - if None, use stdin
    if input_file is None:
        input_file = "-"  # format_utils treats "-" as stdin

    objs = load_objects(input_file)
    import pandas as pd
    df = pd.DataFrame(objs)
    import matplotlib.pyplot as plt

    if not y_column:
        df[x_column].value_counts().plot(kind='bar', figsize=(width, height))
    else:
        df.groupby(x_column)[y_column].value_counts().unstack().plot(kind='bar', figsize=(width, height))
    plt.title(title)
    plt.ylabel(x_column.replace('_', ' ').title())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight", dpi=150)
    plt.close()


@plot_cli.command()
@click.argument("input_file", required=False)
@click.option("--x-column", "-x", required=True, help="Column to use for x-axis")
@click.option("--y-column", "-y", required=True, help="Column to use for y-axis")
@click.option("--title", "-t", help="Title for the heatmap")
@click.option("--width", "-w", type=int, default=10, show_default=True, help="Width of the figure in inches")
@click.option("--height", "-h", type=int, default=8, show_default=True, help="Height of the figure in inches")
@click.option("--output", "-o", required=True, help="Output file path")
def diverging_barchart(input_file: Optional[str], x_column: str, y_column: str, title: Optional[str], width: int, height: int, output: str):
    """
    Create a diverging barchart from a tabular data file.

    The x-axis is the score, and the y-axis is the y_column.
    The bars are colored red if the score is negative, and green if the score is positive.
    The bars are annotated with the score value.
    The bars are sorted by the score value.
    The bars are centered on the score value.
    """
    # Handle file path - if None, use stdin
    if input_file is None:
        input_file = "-"  # format_utils treats "-" as stdin

    objs = load_objects(input_file)
    import pandas as pd
    df = pd.DataFrame(objs)
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Calculate appropriate figure height based on number of rows
    num_rows = len(df)
    calculated_height = max(height, num_rows * 0.4)  # At least 0.4 inches per row
    
    plt.figure(figsize=(width, calculated_height))

    # Create color palette based on actual values
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in df[x_column]]
    
    # Create the plot using seaborn with explicit color mapping
    ax = sns.barplot(data=df, y=y_column, x=x_column, palette=colors, order=df[y_column])

    # Add vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8, zorder=10)

    # Customize
    plt.xlabel('Score', fontsize=12, fontweight='bold')
    plt.ylabel('Tasks', fontsize=12, fontweight='bold')
    
    # Use provided title or default
    plot_title = title if title else 'Task Scores Distribution'
    plt.title(plot_title, fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis limits based on actual data range
    x_min, x_max = df[x_column].min(), df[x_column].max()
    margin = max(0.1, (x_max - x_min) * 0.1)  # 10% margin or 0.1, whichever is larger
    plt.xlim(x_min - margin, x_max + margin)

    # Ensure y-axis labels are not truncated
    plt.subplots_adjust(left=0.3)  # Increase left margin
    
    # Add score annotations using the actual bar positions
    bars = ax.patches
    for i, (bar, score) in enumerate(zip(bars, df[x_column])):
        # Get the actual y-position of the bar center
        y_pos = bar.get_y() + bar.get_height() / 2
        
        # Position text based on score value
        x_offset = 0.02 * (x_max - x_min)  # 2% of data range
        x_pos = score + (x_offset if score >= 0 else -x_offset)
        
        plt.text(x_pos, y_pos, f'{score:.2f}', 
                va='center', ha='left' if score >= 0 else 'right', 
                fontsize=max(8, min(10, 120 / num_rows)), 
                zorder=11)  # Ensure text is on top

    ax.tick_params(axis='y', labelsize=10)
    # Make y-axis labels smaller if there are many rows
    #if num_rows > 20:
    #    ax.tick_params(axis='y', labelsize=max(6, min(10, 200 / num_rows)))
    
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight", dpi=150)
    plt.close()


# lineplot
@plot_cli.command()
@click.argument("input_file", required=False)
@click.option("--x-column", "-x", required=True, help="Column to use for x-axis")
@click.option("--group-by", "-g", required=True, help="Column to group by")
@click.option("--period", "-p", help="Period to group by (e.g. M, Y, Q, W, D)")
@click.option("--exclude", "-E", help="Exclude group-by values (comma-separated)")
@click.option("--title", "-t", help="Title for the heatmap")
@click.option("--minimum-entries", "-m", type=int, default=1, help="Exclude groups with fewer than this number of entries")
@click.option("--output", "-o", required=True, help="Output file path")
def lineplot(input_file: Optional[str], x_column: str, group_by: str, period: str, exclude: Optional[str], minimum_entries: int, title: Optional[str], output: str):
    """
    Create a lineplot from a tabular data file.
    """
    # Handle file path - if None, use stdin
    if input_file is None:
        input_file = "-"  # format_utils treats "-" as stdin

    objs = load_objects(input_file)
    import pandas as pd
    df = pd.DataFrame(objs)

    if exclude:
        exclude_values = exclude.split(',')
        df = df[~df[group_by].isin(exclude_values)]

    if minimum_entries and minimum_entries > 1:
        df = df.groupby(group_by).filter(lambda x: len(x) >= minimum_entries)

    # assume datetime
    if period:
        df[x_column] = pd.to_datetime(df[x_column], errors='coerce')  # Convert invalid to NaT
        if df[x_column].dt.tz is not None:
            df[x_column] = df[x_column].dt.tz_localize(None)
        # Drop NaT values
        df = df.dropna(subset=[x_column])
        df['period'] = df[x_column].dt.to_period(period)
        # Convert period back to timestamp for plotting
        grouped_data = df.groupby(['period', group_by]).size().reset_index(name='count')
        grouped_data['period'] = grouped_data['period'].dt.to_timestamp()
    else:
        grouped_data = df.groupby([group_by]).size().reset_index(name='count')
        grouped_data['period'] = grouped_data[group_by]

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 8))  # Increased size for better readability
    
    # Use a colorblind-friendly palette
    colors = sns.color_palette("colorblind", n_colors=len(grouped_data[group_by].unique()))
    
    # Define line styles for additional differentiation
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    # Create the plot with different styles for each line
    unique_groups = grouped_data[group_by].unique()
    
    for i, group in enumerate(unique_groups):
        group_data = grouped_data[grouped_data[group_by] == group]
        plt.plot(group_data['period'], group_data['count'], 
                label=group,
                color=colors[i % len(colors)],
                linestyle=line_styles[i % len(line_styles)],
                marker=markers[i % len(markers)],
                markersize=6,
                linewidth=2.5,
                markevery=max(1, len(group_data) // 10))  # Show markers at reasonable intervals
    
    # Add direct labels at the end of each line
    for i, group in enumerate(unique_groups):
        group_data = grouped_data[grouped_data[group_by] == group].sort_values('period')
        if len(group_data) > 0:
            last_point = group_data.iloc[-1]
            plt.annotate(group, 
                        xy=(last_point['period'], last_point['count']),
                        xytext=(10, 0), 
                        textcoords='offset points',
                        fontsize=10,
                        fontweight='bold',
                        va='center',
                        color=colors[i % len(colors)])
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel(x_column.replace('_', ' ').title(), fontsize=12)
    
    # Improve legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # Make room for end labels
    
    plt.savefig(output, bbox_inches="tight", dpi=150)
    plt.close()

def calculate_correlation(df: pd.DataFrame, x_column: str, y_column: str) -> float:
    """Calculate the correlation coefficient between two columns."""
    return df[x_column].corr(df[y_column])

# scatterplot
@plot_cli.command()
@click.argument("input_file", required=False)
@click.option("--x-column", "-x", required=True, help="Column to use for x-axis")
@click.option("--y-column", "-y", required=True, help="Column to use for y-axis")
@click.option("--include-correlation", "-c", is_flag=True, help="Include correlation coefficient in the plot, and add a line of best fit")
@click.option("--title", "-t", help="Title for the heatmap")
@click.option("--output", "-o", required=True, help="Output file path")
def scatterplot(input_file: Optional[str], x_column: str, y_column: str, include_correlation: bool, title: Optional[str], output: str):
    """
    Create a scatterplot from a tabular data file.
    """
    # Handle file path - if None, use stdin
    if input_file is None:
        input_file = "-"  # format_utils treats "-" as stdin

    objs = load_objects(input_file)
    import pandas as pd
    df = pd.DataFrame(objs)
    import seaborn as sns
    import matplotlib.pyplot as plt
    correlation = calculate_correlation(df, x_column, y_column)


    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_column, y=y_column, label=f"Correlation: {correlation:.2f}")

    sns.regplot(data=df, x=x_column, y=y_column, label=f"Correlation: {correlation:.2f}")

    plt.title(title)
    plt.ylabel(y_column.replace('_', ' ').title())
    plt.xlabel(x_column.replace('_', ' ').title())
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight", dpi=150)
    plt.close()

@plot_cli.command()
@click.argument("input_file", required=False)
@click.option("--x-column", "-x", required=True, help="Column to use for x-axis")
@click.option("--y-column", "-y", required=True, help="Column to use for y-axis")
@click.option("--group-by", "-g", required=False, help="Column to group by")
@click.option("--title", "-t", help="Title for the heatmap")
@click.option("--output", "-o", required=True, help="Output file path")
def barplot(input_file: Optional[str], x_column: str, y_column: str, group_by: str, title: Optional[str], output: str):
    """
    Create a barplot from a tabular data file.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Handle file path - if None, use stdin
    if input_file is None:
        input_file = "-"  # format_utils treats "-" as stdin

    objs = load_objects(input_file)
    df = pd.DataFrame(objs)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=x_column, y=y_column)
    plt.title(title)
    # save the plot
    plt.savefig(output, bbox_inches="tight", dpi=150)
    plt.close()


@plot_cli.command()
@click.argument("input_file", required=False)
@click.option("--x-column", "-x", required=True, help="Column to use for x-axis")
@click.option("--y-column", "-y", required=True, help="Column to use for y-axis")
@click.option("--y-explode-lists", "-Y", is_flag=True, help="Explode list values in y-column into separate rows")
@click.option("--group-by", "-g", required=False, help="Column to group by")
@click.option("--width", "-w", type=int, default=10, show_default=True, help="Width of the figure in inches")
@click.option("--height", "-h", type=int, default=8, show_default=True, help="Height of the figure in inches")
@click.option("--title", "-t", help="Title for the heatmap")
@click.option("--output", "-o", required=True, help="Output file path")
def boxplot(input_file: Optional[str], x_column: str, y_column: str, y_explode_lists: bool, group_by: str, width: int, height: int, title: Optional[str], output: str):
    """
    Create a boxplot from a tabular data file.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Handle file path - if None, use stdin
    if input_file is None:
        input_file = "-"  # format_utils treats "-" as stdin

    objs = load_objects(input_file)
    df = pd.DataFrame(objs)

    print("Y COLUMN", type(df[y_column].iloc[0]), isinstance(df[y_column].iloc[0], (list, tuple)), df[y_column].head())
    print("X COLUMN", df[x_column].head())

    # if y column is a list, join or explode it
    if isinstance(df[y_column].iloc[0], (list, tuple)):
        if y_explode_lists:
            # Explode the list into separate rows
            df = df.explode(y_column).reset_index(drop=True)
        else:
            # Join the list elements into a single string
            df[y_column] = df[y_column].apply(lambda x: ",".join(x or []) if isinstance(x, (list, tuple)) else x)
    if isinstance(df[x_column].iloc[0], (list, tuple)):
        df[x_column] = df[x_column].apply(lambda x: ",".join(x or []) if isinstance(x, (list, tuple)) else x)

    # sort the dataframe by the x_column
    df = df.sort_values(by=x_column, ascending=False)

    # Define the desired order for your ranges
    # range_order = sorted(df[x_column].unique())

    plt.figure(figsize=(width, height))
    sns.catplot(data=df, x=x_column, y=y_column, hue=group_by, kind="box", 
                height=height, aspect=width/height)
    #sns.boxplot(data=df, x=x_column, y=y_column, hue=group_by, order=range_order)
    plt.title(title)
    # save the plot
    plt.savefig(output, bbox_inches="tight", dpi=150)
    plt.close()


@plot_cli.command()
@click.argument("input_file", required=False)
@click.option("--title", "-t", help="Title for the heatmap")
@click.option("--width", "-w", type=int, default=10, show_default=True, help="Width of the figure in inches")
@click.option("--height", "-h", type=int, default=8, show_default=True, help="Height of the figure in inches")
@click.option("--output", "-o", required=True, help="Output file path")
def facet_chart(
    input_file: Optional[str],
    title: Optional[str],
    width: int,
    height: int,
    output: str,
):
    """
    Create a facet chart from a tabular data file.
    """
    # Handle file path - if None, use stdin
    if input_file is None:
        input_file = "-"  # format_utils treats "-" as stdin

    objs = load_objects(input_file)
    if len(objs) != 1:
        raise ValueError("Facet chart requires exactly one object")
    
    from linkml_store.plotting.facet_chart import create_faceted_horizontal_barchart
    create_faceted_horizontal_barchart(objs[0], output)
    click.echo(f"Facet chart saved to {output}")


@plot_cli.command()
@click.pass_context
@click.option("--collections", "-c", help="Comma-separated list of collection names", required=True)
@click.option("--method", "-m", type=click.Choice(["umap", "tsne", "pca"]), default="tsne", help="Reduction method")
@click.option("--index-name", "-i", help="Name of index to use (defaults to first available)")
@click.option("--color-field", help="Field to use for coloring points")
@click.option("--shape-field", default="collection", help="Field to use for point shapes")
@click.option("--size-field", help="Field to use for point sizes")
@click.option("--hover-fields", help="Comma-separated list of fields to show on hover")
@click.option("--limit-per-collection", "-l", type=int, help="Max embeddings per collection")
@click.option("--n-neighbors", type=int, default=15, help="UMAP n_neighbors parameter")
@click.option("--min-dist", type=float, default=0.1, help="UMAP min_dist parameter")
@click.option("--perplexity", type=float, default=30.0, help="t-SNE perplexity parameter")
@click.option("--random-state", type=int, default=42, help="Random seed for reproducibility")
@click.option("--width", type=int, default=800, help="Plot width in pixels")
@click.option("--height", type=int, default=600, help="Plot height in pixels")
@click.option("--dark-mode/--no-dark-mode", default=False, help="Use dark mode theme")
@click.option("--output", "-o", type=click.Path(), help="Output HTML file path")
def multi_collection_embeddings(ctx, collections, method, index_name, color_field, shape_field,
                   size_field, hover_fields, limit_per_collection, n_neighbors,
                   min_dist, perplexity, random_state, width, height, dark_mode, output):
    """
    Create an interactive plot of embeddings from indexed collections.

    Example:
        linkml-store -d mydb.ddb plot multi-collection-embeddings --collections coll1,coll2 --method umap -o plot.html
    """
    from linkml_store.utils.embedding_utils import extract_embeddings_from_multiple_collections
    from linkml_store.plotting.dimensionality_reduction import reduce_dimensions
    from linkml_store.plotting.embedding_plot import plot_embeddings as create_plot, EmbeddingPlotConfig

    # Parse collections
    collection_names = [c.strip() for c in collections.split(",")]

    # Parse hover fields
    hover_field_list = []
    if hover_fields:
        hover_field_list = [f.strip() for f in hover_fields.split(",")]

    # Extract embeddings
    db = ctx.obj["settings"].database
    click.echo(f"Extracting embeddings from collections: {collection_names}")

    embedding_data = extract_embeddings_from_multiple_collections(
        database=db,
        collection_names=collection_names,
        index_name=index_name,
        limit_per_collection=limit_per_collection,
        include_metadata=True,
        normalize=True
    )

    click.echo(f"Extracted {embedding_data.n_samples} embeddings with {embedding_data.n_dimensions} dimensions")

    # Validate embeddings before reduction
    from linkml_store.plotting.dimensionality_reduction import validate_embeddings
    validation_results = validate_embeddings(embedding_data.vectors)

    if validation_results["warnings"]:
        click.echo("Embedding validation warnings:", err=True)
        for warning in validation_results["warnings"]:
            click.echo(f"  - {warning}", err=True)

    # Log detailed stats for debugging
    logger.info(f"Embedding validation results: {validation_results}")

    # Perform dimensionality reduction
    click.echo(f"Performing {method.upper()} dimensionality reduction...")

    # Set method-specific parameters
    reduction_params = {
        "n_components": 2,
        "random_state": random_state
    }

    if method == "umap":
        reduction_params.update({
            "n_neighbors": min(n_neighbors, embedding_data.n_samples - 1),
            "min_dist": min_dist
        })
    elif method == "tsne":
        reduction_params["perplexity"] = min(perplexity, embedding_data.n_samples / 4)

    try:
        reduction_result = reduce_dimensions(
            embedding_data.vectors,
            method=method,
            **reduction_params
        )
    except ImportError as e:
        click.echo(f"Missing dependency: {e}", err=True)
        click.echo("Install with: pip install umap-learn scikit-learn", err=True)
        return
    except Exception as e:
        click.echo(f"Error during dimensionality reduction: {e}", err=True)
        return

    logger.info(f"Reduction result: {reduction_result}")
    # Create plot configuration
    plot_config = EmbeddingPlotConfig(
        color_field=color_field,
        shape_field=shape_field,
        size_field=size_field,
        hover_fields=hover_field_list,
        title=f"Embedding Visualization ({', '.join(collection_names)})",
        width=width,
        height=height,
        dark_mode=dark_mode
    )

    # Create plot
    click.echo("Creating interactive plot...")
    fig = create_plot(
        embedding_data=embedding_data,
        reduction_result=reduction_result,
        config=plot_config,
        output_file=output
    )

    if output:
        click.echo(f"Plot saved to {output}")
    else:
        # If no output file, try to show in browser
        fig.show()
        click.echo("Plot opened in browser")

if __name__ == "__main__":
    plot_cli()