"""
Tests for the heatmap plotting functionality.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from linkml_store.plotting.heatmap import create_heatmap, export_heatmap_data


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    data = {
        "category_x": ["A", "B", "C", "A", "B", "A", "C", "B", "A"],
        "category_y": ["X", "X", "X", "Y", "Y", "Z", "Z", "Z", "Y"],
        "value": [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0],
    }
    return pd.DataFrame(data)


def test_create_heatmap(sample_data):
    """Test creating a heatmap."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "test_heatmap.png")
        
        # Test basic heatmap creation
        fig, ax = create_heatmap(
            data=sample_data,
            x_column="category_x",
            y_column="category_y",
            value_column="value",
            title="Test Heatmap",
            output_file=output_file
        )
        
        # Check the output file was created
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0
        
        # Test without value column (should use frequency counts)
        fig, ax = create_heatmap(
            data=sample_data,
            x_column="category_x",
            y_column="category_y",
            output_file=None  # Don't save to file
        )
        
        # Check that the axes were created properly
        assert ax.get_xlabel() == "category_x"
        assert "category_y" in ax.get_ylabel()
        
        # Test with duplicate removal
        duplicated_data = sample_data.copy()
        # Add duplicate row
        duplicated_data = pd.concat([duplicated_data, sample_data.iloc[0:1]])
        
        # Create heatmap with duplicates removed
        fig, ax = create_heatmap(
            data=duplicated_data,
            x_column="category_x",
            y_column="category_y",
            value_column="value",
            remove_duplicates=True
        )
        
        # Create heatmap without removing duplicates 
        fig, ax = create_heatmap(
            data=duplicated_data,
            x_column="category_x",
            y_column="category_y",
            value_column="value",
            remove_duplicates=False
        )


def test_export_heatmap_data(sample_data):
    """Test exporting heatmap data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test exporting with value column
        output_file_1 = os.path.join(temp_dir, "heatmap_data_with_values.csv")
        result_df_1 = export_heatmap_data(
            data=sample_data,
            x_column="category_x",
            y_column="category_y",
            value_column="value",
            output_file=output_file_1
        )
        
        # Check output file exists and has content
        assert os.path.exists(output_file_1)
        assert os.path.getsize(output_file_1) > 0
        
        # Check the dataframe has the right structure
        assert "category_y" in result_df_1.columns
        assert "A" in result_df_1.columns
        assert "B" in result_df_1.columns
        assert "C" in result_df_1.columns
        
        # Test exporting without value column (frequency counts)
        output_file_2 = os.path.join(temp_dir, "heatmap_data_counts.csv")
        result_df_2 = export_heatmap_data(
            data=sample_data,
            x_column="category_x",
            y_column="category_y",
            output_file=output_file_2
        )
        
        # Check output file exists and has content
        assert os.path.exists(output_file_2)
        assert os.path.getsize(output_file_2) > 0
        
        # Verify the counts
        # X row should have 1 for A, 1 for B, 1 for C
        x_row = result_df_2[result_df_2["category_y"] == "X"]
        assert x_row["A"].iloc[0] == 1
        assert x_row["B"].iloc[0] == 1
        assert x_row["C"].iloc[0] == 1
        
        # Test with duplicates
        duplicated_data = sample_data.copy()
        # Add duplicate row for A/X
        duplicated_data = pd.concat([duplicated_data, sample_data.iloc[0:1]])
        
        # Export with remove_duplicates=True
        output_file_3 = os.path.join(temp_dir, "heatmap_data_no_duplicates.csv")
        result_df_3 = export_heatmap_data(
            data=duplicated_data,
            x_column="category_x",
            y_column="category_y",
            value_column="value",
            output_file=output_file_3,
            remove_duplicates=True
        )
        
        # Check that duplicates are removed
        assert len(result_df_3) == 3  # One row per y category
        
        # Export with remove_duplicates=False 
        output_file_4 = os.path.join(temp_dir, "heatmap_data_with_duplicates.csv")
        result_df_4 = export_heatmap_data(
            data=duplicated_data,
            x_column="category_x",
            y_column="category_y",
            value_column="value",
            output_file=output_file_4,
            remove_duplicates=False
        )
        
        # Should still have the same number of rows since pivot tables 
        # automatically handle duplicates by aggregating
        assert len(result_df_4) == 3


def test_heatmap_input_validation(sample_data):
    """Test input validation for heatmap creation."""
    # Test with invalid x_column
    with pytest.raises(ValueError, match="x_column 'invalid_column' not found"):
        create_heatmap(
            data=sample_data,
            x_column="invalid_column",
            y_column="category_y"
        )
    
    # Test with invalid y_column
    with pytest.raises(ValueError, match="y_column 'invalid_column' not found"):
        create_heatmap(
            data=sample_data,
            x_column="category_x",
            y_column="invalid_column"
        )
    
    # Test with invalid value_column
    with pytest.raises(ValueError, match="value_column 'invalid_column' not found"):
        create_heatmap(
            data=sample_data,
            x_column="category_x",
            y_column="category_y",
            value_column="invalid_column"
        )