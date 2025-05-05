"""
Heatmap visualization module for LinkML data.

This module provides functions to generate heatmaps from pandas DataFrames or tabular data files.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster import hierarchy
from scipy.spatial import distance

from linkml_store.utils.format_utils import Format, load_objects, write_output

logger = logging.getLogger(__name__)


def create_heatmap(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    value_column: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: Union[str, LinearSegmentedColormap] = "YlGnBu",
    annot: bool = True,
    fmt: Optional[str] = None,  # Dynamically determined based on data
    linewidths: float = 0.5,
    linecolor: str = "white",
    square: bool = False,
    output_file: Optional[str] = None,
    dpi: int = 300,
    missing_value: Any = np.nan,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    robust: bool = False,
    remove_duplicates: bool = True,
    font_size: int = 10,
    cluster: Union[bool, Literal["both", "x", "y"]] = False,
    cluster_method: str = "complete",  # linkage method: complete, average, single, etc.
    cluster_metric: str = "euclidean",  # distance metric: euclidean, cosine, etc.
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a heatmap from a pandas DataFrame.

    Args:
        data: Input DataFrame containing the data to plot
        x_column: Column to use for x-axis categories
        y_column: Column to use for y-axis categories
        value_column: Column containing values for the heatmap. If None, frequency counts will be used.
        title: Title for the heatmap
        figsize: Figure size as (width, height) in inches
        cmap: Colormap for the heatmap
        annot: Whether to annotate cells with values
        fmt: String formatting code for annotations (auto-detected if None)
        linewidths: Width of lines between cells
        linecolor: Color of lines between cells
        square: Whether to make cells square
        output_file: File path to save the figure (optional)
        dpi: Resolution for saved figure
        missing_value: Value to use for missing data (defaults to NaN)
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        robust: If True, compute colormap limits using robust quantiles instead of min/max
        remove_duplicates: If True, removes duplicate rows before creating the heatmap
        font_size: Font size for annotations
        cluster: Whether and which axes to cluster:
                - False: No clustering (default)
                - True or "both": Cluster both x and y axes
                - "x": Cluster only x-axis
                - "y": Cluster only y-axis
        cluster_method: Linkage method for hierarchical clustering
                      (e.g., "single", "complete", "average", "ward")
        cluster_metric: Distance metric for clustering (e.g., "euclidean", "correlation", "cosine")
        **kwargs: Additional keyword arguments to pass to seaborn's heatmap function

    Returns:
        Tuple containing the figure and axes objects
    """
    # Validate input
    if x_column not in data.columns:
        raise ValueError(f"x_column '{x_column}' not found in DataFrame columns: {list(data.columns)}")
    if y_column not in data.columns:
        raise ValueError(f"y_column '{y_column}' not found in DataFrame columns: {list(data.columns)}")
    if value_column and value_column not in data.columns:
        raise ValueError(f"value_column '{value_column}' not found in DataFrame columns: {list(data.columns)}")

    # Remove duplicates by default (assume they're accidents unless user overrides)
    if remove_duplicates:
        data = data.drop_duplicates()
    
    # Prepare the data
    if value_column:
        # Use the provided value column
        pivot_data = data.pivot_table(
            index=y_column, 
            columns=x_column, 
            values=value_column, 
            aggfunc='mean',
            fill_value=missing_value
        )
    else:
        # Use frequency counts
        cross_tab = pd.crosstab(data[y_column], data[x_column])
        pivot_data = cross_tab
    
    # Auto-detect format string if not provided
    if fmt is None:
        # Check if the pivot table contains integers only
        if pivot_data.dtypes.apply(lambda x: pd.api.types.is_integer_dtype(x)).all():
            fmt = 'd'  # Integer format
        else:
            fmt = '.1f'  # One decimal place for floats
    
    # Make sure all cells have a reasonable minimum size
    min_height = max(4, 80 / len(pivot_data.index) if len(pivot_data.index) > 0 else 10)
    min_width = max(4, 80 / len(pivot_data.columns) if len(pivot_data.columns) > 0 else 10)
    
    # Adjust figure size based on the number of rows and columns 
    adjusted_height = max(figsize[1], min_height * len(pivot_data.index) / 10)
    adjusted_width = max(figsize[0], min_width * len(pivot_data.columns) / 10)
    adjusted_figsize = (adjusted_width, adjusted_height)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=adjusted_figsize)
    
    # Apply clustering if requested
    row_linkage = None
    col_linkage = None
    
    if cluster:
        cluster_axes = cluster
        if cluster_axes is True:
            cluster_axes = "both"
            
        # Fill NAs for clustering
        pivot_data_for_clustering = pivot_data.fillna(0)
        
        # Cluster rows (y-axis)
        if cluster_axes in ["both", "y"]:
            try:
                # Calculate distance matrix and linkage for rows
                row_distances = distance.pdist(pivot_data_for_clustering.values, metric=cluster_metric)
                row_linkage = hierarchy.linkage(row_distances, method=cluster_method)
                
                # Reorder rows based on clustering
                row_dendrogram = hierarchy.dendrogram(row_linkage, no_plot=True)
                row_order = row_dendrogram['leaves']
                pivot_data = pivot_data.iloc[row_order]
                
                logger.info(f"Applied clustering to rows using {cluster_method} linkage and {cluster_metric} metric")
            except Exception as e:
                logger.warning(f"Failed to cluster rows: {e}")
        
        # Cluster columns (x-axis)
        if cluster_axes in ["both", "x"]:
            try:
                # Calculate distance matrix and linkage for columns
                col_distances = distance.pdist(pivot_data_for_clustering.values.T, metric=cluster_metric)
                col_linkage = hierarchy.linkage(col_distances, method=cluster_method)
                
                # Reorder columns based on clustering
                col_dendrogram = hierarchy.dendrogram(col_linkage, no_plot=True)
                col_order = col_dendrogram['leaves']
                pivot_data = pivot_data.iloc[:, col_order]
                
                logger.info(f"Applied clustering to columns using {cluster_method} linkage and {cluster_metric} metric")
            except Exception as e:
                logger.warning(f"Failed to cluster columns: {e}")

    # Create the heatmap
    sns.heatmap(
        pivot_data,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        linewidths=linewidths,
        linecolor=linecolor,
        square=square,
        vmin=vmin,
        vmax=vmax,
        robust=robust,
        ax=ax,
        annot_kws={'fontsize': font_size},
        **kwargs
    )
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=font_size + 4)
    
    # Improve display of tick labels
    plt.xticks(rotation=45, ha="right", fontsize=font_size)
    plt.yticks(rotation=0, fontsize=font_size)
    
    # Add grid lines to make the table more readable
    ax.grid(False)
    
    # Improve contrast for better readability
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if output file is specified
    if output_file:
        output_path = Path(output_file)
        output_dir = output_path.parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        logger.info(f"Heatmap saved to {output_file}")
    
    return fig, ax


def heatmap_from_file(
    file_path: str,
    x_column: str,
    y_column: str,
    value_column: Optional[str] = None,
    format: Optional[Union[Format, str]] = None,
    compression: Optional[str] = None,
    output_file: Optional[str] = None,
    remove_duplicates: bool = True,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a heatmap from a file (CSV, TSV, etc.).

    Args:
        file_path: Path to the input file or "-" for stdin
        x_column: Column to use for x-axis categories
        y_column: Column to use for y-axis categories
        value_column: Column containing values for the heatmap. If None, frequency counts will be used.
        format: Format of the input file (auto-detected if None)
        compression: Compression format ('gz' or 'tgz')
        output_file: File path to save the figure (optional)
        remove_duplicates: If True, removes duplicate rows before creating the heatmap
        **kwargs: Additional arguments to pass to create_heatmap

    Returns:
        Tuple containing the figure and axes objects
    """
    # Handle stdin input safely
    import sys
    import io
    import pandas as pd
    import click
    
    # Load the data
    if file_path == "-":
        # Read directly from stdin since format_utils will use sys.stdin which may already be consumed
        if not format or str(format).lower() in ['csv', 'tsv']:
            # Default to CSV if no format specified
            delimiter = ',' if not format or str(format).lower() == 'csv' else '\t'
            df = pd.read_csv(sys.stdin, delimiter=delimiter)
        else:
            # Try to use format_utils but with a backup plan
            try:
                objs = load_objects(file_path, format=format, compression=compression)
                df = pd.DataFrame(objs)
            except ValueError as e:
                if "I/O operation on closed file" in str(e):
                    logger.warning("Could not read from stdin. It may have been consumed already.")
                    raise click.UsageError("Error reading from stdin. Please provide a file path or ensure stdin has data.")
                else:
                    raise
    else:
        # For regular files, use format_utils as normal
        if (not format or format in ["csv", "tsv"]) and not compression:
            df = pd.read_csv(file_path)
        else:
            objs = load_objects(file_path, format=format, compression=compression)
            df = pd.DataFrame(objs)

    # Create the heatmap
    return create_heatmap(
        data=df,
        x_column=x_column,
        y_column=y_column,
        value_column=value_column,
        output_file=output_file,
        remove_duplicates=remove_duplicates,
        **kwargs
    )


def export_heatmap_data(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    value_column: Optional[str] = None,
    output_file: Optional[str] = None,
    format: Union[Format, str] = Format.CSV,
    missing_value: Any = np.nan,
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Export heatmap data to a file or return it as a DataFrame.

    Args:
        data: Input DataFrame containing the data
        x_column: Column to use for x-axis categories
        y_column: Column to use for y-axis categories
        value_column: Column containing values for the heatmap. If None, frequency counts will be used.
        output_file: File path to save the data (optional)
        format: Output format for the file
        missing_value: Value to use for missing data
        remove_duplicates: If True, removes duplicate rows before creating the pivot table

    Returns:
        DataFrame containing the pivot table data
    """
    # Remove duplicates by default (assume they're accidents unless user overrides)
    if remove_duplicates:
        # Keep the first occurrence of each x_column, y_column combination
        data = data.drop_duplicates(subset=[x_column, y_column])

    # Prepare the data
    if value_column:
        # Use the provided value column
        pivot_data = data.pivot_table(
            index=y_column, 
            columns=x_column, 
            values=value_column, 
            aggfunc='mean',
            fill_value=missing_value
        )
    else:
        # Use frequency counts
        cross_tab = pd.crosstab(data[y_column], data[x_column])
        pivot_data = cross_tab
    
    # Reset index to make the y_column a regular column
    result_df = pivot_data.reset_index()
    
    # Write to file if output_file is provided
    if output_file:
        # Convert to records format for writing
        records = result_df.to_dict(orient='records')
        write_output(records, format=format, target=output_file)
        logger.info(f"Heatmap data saved to {output_file}")
    
    return result_df