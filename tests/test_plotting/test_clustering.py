"""
Tests for the clustering functionality in heatmap visualization.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from linkml_store.plotting.heatmap import create_heatmap


@pytest.fixture
def cluster_test_data():
    """Create a sample DataFrame designed for clustering tests."""
    # Create a dataset with clear patterns for clustering
    np.random.seed(42)  # For reproducibility
    
    # Create 20 rows x 15 columns with patterns
    n_rows = 20
    n_cols = 15
    
    # Base data - random
    data = np.random.rand(n_rows, n_cols)
    
    # Create patterns in the data
    # First 5 rows have similar pattern
    data[0:5, :] = data[0:5, :] * 0.5 + 0.8
    # Next 5 rows have another pattern
    data[5:10, :] = data[5:10, :] * 0.3 + 0.2
    # Next 5 rows have a third pattern
    data[10:15, :] = data[10:15, :] * 0.8 + 0.1
    # Last 5 rows have a fourth pattern
    data[15:20, :] = data[15:20, :] * 0.4 + 0.5
    
    # Similarly, create patterns in the columns
    data[:, 0:5] = data[:, 0:5] * 1.2
    data[:, 5:10] = data[:, 5:10] * 0.8
    data[:, 10:15] = data[:, 10:15] * 1.5
    
    # Convert to DataFrame
    row_names = [f"row_{i}" for i in range(n_rows)]
    col_names = [f"col_{i}" for i in range(n_cols)]
    df = pd.DataFrame(data, columns=col_names)
    
    # Add row and column identifiers
    df["row_id"] = row_names
    
    return df


def test_clustering_options(cluster_test_data):
    """Test that clustering options work as expected."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with no clustering (default)
        fig1, ax1 = create_heatmap(
            data=cluster_test_data,
            x_column="col_0",  # Any column name will work for this test
            y_column="row_id",
            value_column="col_1",  # Use another column as values
            cluster=False
        )
        
        # Test with clustering both axes
        fig2, ax2 = create_heatmap(
            data=cluster_test_data,
            x_column="col_0",
            y_column="row_id",
            value_column="col_1",
            cluster="both",
            cluster_method="complete",
            cluster_metric="euclidean"
        )
        
        # Test with clustering only x-axis
        fig3, ax3 = create_heatmap(
            data=cluster_test_data,
            x_column="col_0",
            y_column="row_id",
            value_column="col_1",
            cluster="x"
        )
        
        # Test with clustering only y-axis
        fig4, ax4 = create_heatmap(
            data=cluster_test_data,
            x_column="col_0",
            y_column="row_id",
            value_column="col_1",
            cluster="y"
        )
        
        # Test different linkage methods
        fig5, ax5 = create_heatmap(
            data=cluster_test_data,
            x_column="col_0",
            y_column="row_id",
            value_column="col_1",
            cluster="both",
            cluster_method="single"
        )
        
        # Test different distance metrics
        fig6, ax6 = create_heatmap(
            data=cluster_test_data,
            x_column="col_0",
            y_column="row_id",
            value_column="col_1",
            cluster="both",
            cluster_metric="correlation"
        )
        
        # Verify that all axes were created
        assert ax1 is not None
        assert ax2 is not None
        assert ax3 is not None
        assert ax4 is not None
        assert ax5 is not None
        assert ax6 is not None
        
        # Close all figures to avoid memory leaks
        import matplotlib.pyplot as plt
        plt.close('all')