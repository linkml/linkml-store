from collections import Counter

import pandas as pd
import pytest

from linkml_store.utils.enrichment_analyzer import (
    EnrichmentAnalyzer,
)  # Assuming the previous code is in enrichment_analysis.py


@pytest.fixture
def sample_df():
    """Create a test DataFrame with known enrichment patterns"""
    data = {
        "sample_id": [
            "sample1",
            "sample1",
            "sample1",
            "sample1",
            "sample1",
            "sample2",
            "sample2",
            "sample2",
            "sample3",
            "sample3",
            "sample3",
        ],
        "categories": [
            ["A", "B"],
            ["A", "C"],
            ["A", "B"],
            ["B", "C"],
            ["A"],
            ["C", "D"],
            ["C", "D"],
            ["D", "E"],
            ["E", "F"],
            ["E", "F"],
            ["F", "G"],
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def analyzer(sample_df):
    """Create an EnrichmentAnalyzer instance with the sample data"""
    return EnrichmentAnalyzer(sample_df, "sample_id", "categories")


def test_initialization(analyzer, sample_df):
    """Test that the analyzer initializes correctly"""
    assert analyzer.df.equals(sample_df)
    assert analyzer.sample_key == "sample_id"
    assert analyzer.classification_key == "categories"
    assert isinstance(analyzer.global_stats, Counter)
    assert len(analyzer.sample_cache) == 0


def test_global_stats_computation(analyzer):
    """Test that global statistics are computed correctly"""
    expected_counts = {"A": 4, "B": 3, "C": 4, "D": 3, "E": 3, "F": 3, "G": 1}
    assert dict(analyzer.global_stats) == expected_counts


def test_sample_stats_computation(analyzer):
    """Test that sample-specific statistics are computed correctly"""
    sample1_stats = analyzer._get_sample_stats("sample1")
    expected_sample1 = {"A": 4, "B": 3, "C": 2}
    assert dict(sample1_stats) == expected_sample1

    # Test caching
    assert "sample1" in analyzer.sample_cache
    assert dict(analyzer.sample_cache["sample1"]) == expected_sample1


def test_enrichment_analysis(analyzer):
    """Test the enrichment analysis results with different multiple testing corrections"""

    # Test without correction
    enriched_none = analyzer.find_enriched_categories(
        "sample1", min_occurrences=2, p_value_threshold=0.05, multiple_testing_correction="none"
    )

    # Test with Bonferroni correction
    enriched_bonf = analyzer.find_enriched_categories(
        "sample1", min_occurrences=2, p_value_threshold=0.05, multiple_testing_correction="bonf"
    )

    # Test with Benjamini-Hochberg correction
    enriched_bh = analyzer.find_enriched_categories(
        "sample1", min_occurrences=2, p_value_threshold=0.05, multiple_testing_correction="bh"
    )

    # Convert results to more easily testable format
    enriched_dict_none = {result.category: result for result in enriched_none}
    enriched_dict_bonf = {result.category: result for result in enriched_bonf}
    enriched_dict_bh = {result.category: result for result in enriched_bh}

    # Check that corrections are working as expected
    assert len(enriched_none) >= len(enriched_bonf)  # Bonferroni should be most conservative
    assert len(enriched_bh) >= len(enriched_bonf)  # BH should find more than Bonferroni

    # Check that A and B are enriched in at least one method
    assert any(("A" in d) for d in [enriched_dict_none, enriched_dict_bonf, enriched_dict_bh])

    # Check fold changes make sense
    for enriched_dict in [enriched_dict_none, enriched_dict_bonf, enriched_dict_bh]:
        if "A" in enriched_dict:
            result = enriched_dict["A"]
            assert result.fold_change > 1.0  # Should be enriched

    # Check p-values and adjusted p-values are valid
    for enriched_dict in [enriched_dict_none, enriched_dict_bonf, enriched_dict_bh]:
        for result in enriched_dict.values():
            assert 0 <= result.original_p_value <= 1
            assert 0 <= result.adjusted_p_value <= 1
            assert (
                result.adjusted_p_value >= result.original_p_value
            )  # Adjusted p-value should never be smaller than original


def test_edge_cases(sample_df):
    """Test edge cases and potential error conditions"""

    # Test empty DataFrame
    empty_df = pd.DataFrame({"sample_id": [], "categories": []})
    analyzer_empty = EnrichmentAnalyzer(empty_df, "sample_id", "categories")
    assert len(analyzer_empty.global_stats) == 0

    # Test single category
    single_cat_data = {"sample_id": ["sample1", "sample2"], "categories": [["A"], ["A"]]}
    single_cat_df = pd.DataFrame(single_cat_data)
    analyzer_single = EnrichmentAnalyzer(single_cat_df, "sample_id", "categories")
    assert dict(analyzer_single.global_stats) == {"A": 2}

    # Test non-list categories (string input)
    string_cat_data = {"sample_id": ["sample1", "sample2"], "categories": ["A", "B"]}
    string_cat_df = pd.DataFrame(string_cat_data)
    analyzer_string = EnrichmentAnalyzer(string_cat_df, "sample_id", "categories")
    assert dict(analyzer_string.global_stats) == {"A": 1, "B": 1}


def test_invalid_sample_id(analyzer):
    """Test behavior with invalid sample ID"""
    with pytest.raises(KeyError):
        analyzer._get_sample_stats("nonexistent_sample")


def test_min_occurrences_filter(analyzer):
    """Test that minimum occurrences filter works"""
    # Set high minimum occurrences to filter out most categories
    enriched = analyzer.find_enriched_categories("sample1", min_occurrences=10)
    assert len(enriched) == 0  # No categories should meet this threshold


def test_p_value_threshold(analyzer):
    """Test that p-value threshold works"""
    # Set very strict p-value threshold
    strict_enriched = analyzer.find_enriched_categories("sample1", p_value_threshold=0.0001)
    # Set loose p-value threshold
    loose_enriched = analyzer.find_enriched_categories("sample1", p_value_threshold=0.5)

    # Should find more enriched categories with looser threshold
    assert len(strict_enriched) <= len(loose_enriched)


def test_result_sorting(analyzer):
    """Test that results are properly sorted by p-value"""
    enriched = analyzer.find_enriched_categories("sample1")
    p_values = [p for _, _, p in enriched]
    assert p_values == sorted(p_values)  # Should be sorted in ascending order


if __name__ == "__main__":
    pytest.main([__file__])
