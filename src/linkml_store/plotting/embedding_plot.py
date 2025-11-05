"""Plotting utilities for embedding visualizations."""

import logging
from typing import Dict, List, Optional, Union, Literal, Tuple
import numpy as np
from dataclasses import dataclass, field
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from linkml_store.utils.embedding_utils import EmbeddingData
from linkml_store.plotting.dimensionality_reduction import ReductionResult

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingPlotConfig:
    """Configuration for embedding plots."""

    # Visual encoding
    color_field: Optional[str] = None
    shape_field: Optional[str] = "collection"
    size_field: Optional[str] = None
    hover_fields: List[str] = field(default_factory=list)

    # Plot styling
    title: str = "Embedding Visualization"
    width: int = 800
    height: int = 600
    point_size: int = 8
    opacity: float = 0.7

    # Color schemes
    color_discrete_map: Optional[Dict] = None
    color_continuous_scale: str = "Viridis"

    # Shape mapping
    shape_map: Optional[Dict] = None
    marker_symbols: List[str] = field(default_factory=lambda: [
        "circle", "square", "diamond", "cross", "x",
        "triangle-up", "triangle-down", "pentagon", "hexagon", "star"
    ])

    # Display options
    show_legend: bool = True
    show_axes: bool = True
    dark_mode: bool = False


def plot_embeddings(
    embedding_data: EmbeddingData,
    reduction_result: ReductionResult,
    config: Optional[EmbeddingPlotConfig] = None,
    output_file: Optional[str] = None
) -> go.Figure:
    """
    Create interactive plot of embeddings.

    Args:
        embedding_data: Embedding data with metadata
        reduction_result: Dimensionality reduction results
        config: Plot configuration
        output_file: Optional path to save HTML file

    Returns:
        Plotly figure object
    """
    if config is None:
        config = EmbeddingPlotConfig()

    # Prepare data for plotting
    plot_data = _prepare_plot_data(embedding_data, reduction_result, config)

    # Create figure
    fig = _create_scatter_plot(plot_data, config)

    # Apply styling
    fig = _style_figure(fig, config, reduction_result)

    # Save if requested
    if output_file:
        fig.write_html(output_file)
        logger.info(f"Saved plot to {output_file}")

    return fig


def _prepare_plot_data(
    embedding_data: EmbeddingData,
    reduction_result: ReductionResult,
    config: EmbeddingPlotConfig
) -> Dict:
    """Prepare data for plotting."""
    data = {
        "x": reduction_result.coordinates[:, 0],
        "y": reduction_result.coordinates[:, 1],
        "ids": embedding_data.object_ids,
        "collection": embedding_data.collection_names,
    }

    # Add metadata fields
    for field in config.hover_fields:
        values = embedding_data.get_metadata_values(field)
        if values:
            data[field] = values

    # Add color field
    if config.color_field:
        if config.color_field == "collection":
            data["color"] = embedding_data.collection_names
        else:
            data["color"] = embedding_data.get_metadata_values(config.color_field)

    # Add shape field
    if config.shape_field:
        if config.shape_field == "collection":
            data["shape"] = embedding_data.collection_names
        else:
            data["shape"] = embedding_data.get_metadata_values(config.shape_field)

    # Add size field
    if config.size_field:
        data["size"] = embedding_data.get_metadata_values(config.size_field)
    else:
        data["size"] = [config.point_size] * embedding_data.n_samples

    return data


def _create_scatter_plot(
    plot_data: Dict,
    config: EmbeddingPlotConfig
) -> go.Figure:
    """Create the scatter plot."""
    fig = go.Figure()

    # Determine if we need to create separate traces for different shapes
    if "shape" in plot_data and config.shape_field:
        unique_shapes = list(set(plot_data["shape"]))

        # Create shape mapping
        if config.shape_map:
            shape_map = config.shape_map
        else:
            shape_map = {
                shape: config.marker_symbols[i % len(config.marker_symbols)]
                for i, shape in enumerate(unique_shapes)
            }

        # Create trace for each shape category
        for shape_value in unique_shapes:
            mask = [s == shape_value for s in plot_data["shape"]]
            trace_data = {
                key: [v for i, v in enumerate(values) if mask[i]]
                for key, values in plot_data.items()
            }

            # Prepare hover text
            hover_text = _create_hover_text(trace_data, config)

            # Determine color
            if "color" in trace_data:
                color_values = trace_data["color"]
                # Check if categorical or continuous
                if color_values and all(isinstance(v, (int, float)) for v in color_values if v is not None):
                    marker_color = color_values
                    marker_colorscale = config.color_continuous_scale
                else:
                    # Categorical - don't use color map for plotly, let it auto-assign
                    # Just use the raw categorical values
                    marker_color = color_values
                    marker_colorscale = None
            else:
                marker_color = None
                marker_colorscale = None

            # When using separate traces, we can't use categorical colors directly
            # Use a single color per trace instead
            if marker_colorscale is None and marker_color is not None:
                # For categorical, just use the trace name for automatic coloring
                marker_dict = dict(
                    symbol=shape_map.get(shape_value, "circle"),
                    size=trace_data.get("size", config.point_size),
                    opacity=config.opacity,
                    line=dict(width=0.5, color="white")
                )
            else:
                # For continuous colors
                marker_dict = dict(
                    symbol=shape_map.get(shape_value, "circle"),
                    size=trace_data.get("size", config.point_size),
                    color=marker_color,
                    colorscale=marker_colorscale,
                    opacity=config.opacity,
                    line=dict(width=0.5, color="white")
                )

            trace = go.Scatter(
                x=trace_data["x"],
                y=trace_data["y"],
                mode="markers",
                name=str(shape_value),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                marker=marker_dict
            )
            fig.add_trace(trace)
    else:
        # Single trace for all points
        hover_text = _create_hover_text(plot_data, config)

        # Handle colors
        if "color" in plot_data:
            color_values = plot_data["color"]
            if all(isinstance(v, (int, float)) for v in color_values if v is not None):
                marker_color = color_values
                marker_colorscale = config.color_continuous_scale
                showscale = True
            else:
                if config.color_discrete_map:
                    marker_color = [config.color_discrete_map.get(c, c) for c in color_values]
                else:
                    marker_color = color_values
                marker_colorscale = None
                showscale = False
        else:
            marker_color = "blue"
            marker_colorscale = None
            showscale = False

        trace = go.Scatter(
            x=plot_data["x"],
            y=plot_data["y"],
            mode="markers",
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=plot_data.get("size", config.point_size),
                color=marker_color,
                colorscale=marker_colorscale,
                showscale=showscale,
                opacity=config.opacity,
                line=dict(width=0.5, color="white")
            )
        )
        fig.add_trace(trace)

    return fig


def _create_hover_text(
    data: Dict,
    config: EmbeddingPlotConfig
) -> List[str]:
    """Create hover text for each point."""
    hover_texts = []
    n_points = len(data["x"])

    for i in range(n_points):
        lines = []

        # Add ID
        if "ids" in data:
            lines.append(f"<b>ID:</b> {data['ids'][i]}")

        # Add collection
        if "collection" in data:
            lines.append(f"<b>Collection:</b> {data['collection'][i]}")

        # Add hover fields
        for field in config.hover_fields:
            if field in data and data[field][i] is not None:
                lines.append(f"<b>{field}:</b> {data[field][i]}")

        # Add coordinates
        lines.append(f"<b>Coordinates:</b> ({data['x'][i]:.3f}, {data['y'][i]:.3f})")

        hover_texts.append("<br>".join(lines))

    return hover_texts


def _style_figure(
    fig: go.Figure,
    config: EmbeddingPlotConfig,
    reduction_result: ReductionResult
) -> go.Figure:
    """Apply styling to the figure."""
    # Create subtitle with method info
    subtitle = f"Method: {reduction_result.method.upper()}"
    if reduction_result.explained_variance:
        subtitle += f" | Explained variance: {reduction_result.explained_variance:.2%}"

    full_title = f"{config.title}<br><sub>{subtitle}</sub>"

    # Update layout
    layout_updates = dict(
        title=full_title,
        width=config.width,
        height=config.height,
        showlegend=config.show_legend,
        hovermode="closest",
        xaxis=dict(
            title="Component 1",
            showgrid=True,
            gridwidth=1,
            gridcolor="LightGray" if not config.dark_mode else "DarkGray",
            showline=config.show_axes,
            zeroline=True,
        ),
        yaxis=dict(
            title="Component 2",
            showgrid=True,
            gridwidth=1,
            gridcolor="LightGray" if not config.dark_mode else "DarkGray",
            showline=config.show_axes,
            zeroline=True,
        )
    )

    if config.dark_mode:
        layout_updates.update(
            template="plotly_dark",
            paper_bgcolor="black",
            plot_bgcolor="black",
        )
    else:
        layout_updates.update(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

    fig.update_layout(**layout_updates)

    return fig


def plot_embeddings_comparison(
    embedding_datasets: Dict[str, Tuple[EmbeddingData, ReductionResult]],
    config: Optional[EmbeddingPlotConfig] = None,
    output_file: Optional[str] = None
) -> go.Figure:
    """
    Create comparison plot of multiple embedding datasets.

    Args:
        embedding_datasets: Dictionary of (name -> (embedding_data, reduction_result))
        config: Plot configuration
        output_file: Optional path to save HTML file

    Returns:
        Plotly figure with subplots
    """
    if config is None:
        config = EmbeddingPlotConfig()

    n_datasets = len(embedding_datasets)
    n_cols = min(n_datasets, 3)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    # Create subplots
    subplot_titles = list(embedding_datasets.keys())
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    # Add each dataset as a subplot
    for idx, (name, (emb_data, red_result)) in enumerate(embedding_datasets.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        plot_data = _prepare_plot_data(emb_data, red_result, config)

        # Create scatter trace
        hover_text = _create_hover_text(plot_data, config)

        trace = go.Scatter(
            x=plot_data["x"],
            y=plot_data["y"],
            mode="markers",
            name=name,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=plot_data.get("size", config.point_size),
                color=plot_data.get("color", "blue"),
                opacity=config.opacity,
            ),
            showlegend=(idx == 0)  # Only show legend for first subplot
        )

        fig.add_trace(trace, row=row, col=col)

        # Update axes labels
        fig.update_xaxes(title_text="Component 1", row=row, col=col)
        fig.update_yaxes(title_text="Component 2", row=row, col=col)

    # Update overall layout
    fig.update_layout(
        title=config.title,
        width=config.width * n_cols // 2,
        height=config.height * n_rows // 2,
        showlegend=config.show_legend,
        hovermode="closest"
    )

    # Save if requested
    if output_file:
        fig.write_html(output_file)
        logger.info(f"Saved comparison plot to {output_file}")

    return fig


def plot_embedding_clusters(
    embedding_data: EmbeddingData,
    reduction_result: ReductionResult,
    cluster_labels: np.ndarray,
    config: Optional[EmbeddingPlotConfig] = None,
    output_file: Optional[str] = None
) -> go.Figure:
    """
    Plot embeddings with cluster assignments.

    Args:
        embedding_data: Embedding data
        reduction_result: Dimensionality reduction results
        cluster_labels: Cluster labels for each point
        config: Plot configuration
        output_file: Optional path to save HTML file

    Returns:
        Plotly figure with clusters
    """
    if config is None:
        config = EmbeddingPlotConfig()

    # Override color field to use clusters
    config.color_field = "cluster"

    # Add cluster labels to metadata
    for i, label in enumerate(cluster_labels):
        if i < len(embedding_data.metadata):
            embedding_data.metadata[i]["cluster"] = f"Cluster {label}"

    # Create plot
    fig = plot_embeddings(embedding_data, reduction_result, config)

    # Add cluster centroids if possible
    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) < 50:  # Only show centroids for reasonable number of clusters
        for label in unique_labels:
            mask = cluster_labels == label
            centroid_x = np.mean(reduction_result.coordinates[mask, 0])
            centroid_y = np.mean(reduction_result.coordinates[mask, 1])

            fig.add_trace(go.Scatter(
                x=[centroid_x],
                y=[centroid_y],
                mode="markers+text",
                marker=dict(
                    size=15,
                    symbol="star",
                    color="black",
                    line=dict(width=2, color="white")
                ),
                text=f"C{label}",
                textposition="top center",
                showlegend=False,
                hovertemplate=f"Cluster {label} centroid<extra></extra>"
            ))

    # Update title
    fig.update_layout(
        title=f"{config.title}<br><sub>Clusters: {len(unique_labels)}</sub>"
    )

    # Save if requested
    if output_file:
        fig.write_html(output_file)
        logger.info(f"Saved cluster plot to {output_file}")

    return fig