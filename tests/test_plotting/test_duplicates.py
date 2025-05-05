"""
Tests for the duplicate removal functionality in heatmap visualization.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from linkml_store.plotting.heatmap import create_heatmap, export_heatmap_data


@pytest.fixture
def duplicate_test_data():
    """Create a sample DataFrame with duplicates for testing."""
    # Create a dataset with duplicates
    data = {
        "category_x": ["A", "B", "C", "A", "B", "A", "C", "A", "A"],  # Note duplicate A values
        "category_y": ["X", "X", "X", "Y", "Y", "Z", "Z", "X", "Y"],  # Duplicates: A/X and A/Y
        "value": [1, 2, 3, 4, 5, 6, 7, 10, 12],  # Different values for duplicates
    }
    return pd.DataFrame(data)


def test_remove_duplicates_with_value_column(duplicate_test_data):
    """Test that duplicates are properly removed when using a value column."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "heatmap_with_duplicates_removed.png")
        
        # Count initial duplicates
        duplicates = duplicate_test_data.duplicated(subset=["category_x", "category_y"]).sum()
        assert duplicates > 0, "Test data should contain duplicates"
        
        # Test with remove_duplicates=True (default)
        fig1, ax1 = create_heatmap(
            data=duplicate_test_data,
            x_column="category_x",
            y_column="category_y",
            value_column="value",
            output_file=output_file,
            remove_duplicates=True
        )
        
        # Create a temporary file for the data export
        data_file = os.path.join(temp_dir, "data_with_duplicates_removed.csv")
        
        # Export the data with duplicates removed
        result_df = export_heatmap_data(
            data=duplicate_test_data,
            x_column="category_x",
            y_column="category_y",
            value_column="value",
            output_file=data_file,
            remove_duplicates=True
        )
        
        # Check that each x,y combination appears only once in the result
        # Since we're using a pivot table, we check the number of rows (y categories)
        # and columns (x categories)
        unique_x = duplicate_test_data["category_x"].nunique()
        unique_y = duplicate_test_data["category_y"].nunique()
        unique_xy_pairs = len(duplicate_test_data[["category_x", "category_y"]].drop_duplicates())
        
        # The result should have one row per unique y value
        assert len(result_df) == unique_y
        
        # In a pivot table, we get one column per unique x value plus the index column
        assert len(result_df.columns) == unique_x + 1


def test_keep_duplicates(duplicate_test_data):
    """Test that duplicates are properly kept when requested."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "heatmap_with_duplicates_kept.png")
        
        # Test with remove_duplicates=False
        fig2, ax2 = create_heatmap(
            data=duplicate_test_data,
            x_column="category_x",
            y_column="category_y",
            value_column="value",
            output_file=output_file,
            remove_duplicates=False
        )
        
        # Create a copy with the last value for each x-y pair
        expected_data = duplicate_test_data.copy()
        
        # When duplicates are kept, the pivot table aggregates them (default is mean)
        # Get the mean values for each x-y pair
        expected_means = expected_data.groupby(["category_x", "category_y"])["value"].mean()
        
        # Create a temporary file for the data export
        data_file = os.path.join(temp_dir, "data_with_duplicates_kept.csv")
        
        # Export the data with duplicates kept
        result_df = export_heatmap_data(
            data=duplicate_test_data,
            x_column="category_x",
            y_column="category_y",
            value_column="value",
            output_file=data_file,
            remove_duplicates=False
        )
        
        # Check that the file was created
        assert os.path.exists(data_file)
        
        # The pivot table should still have one row per unique y value
        # because duplicates are aggregated, not included separately
        unique_y = duplicate_test_data["category_y"].nunique()
        assert len(result_df) == unique_y