from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats

from linkml_store.api import Collection


class EnrichedCategory(BaseModel):
    """
    Information about a category enriched in a sample
    """

    category: str
    fold_change: float
    original_p_value: float
    adjusted_p_value: float


class EnrichmentAnalyzer:
    def __init__(self, df: pd.DataFrame, sample_key: str, classification_key: str):
        """
        Initialize the analyzer with a DataFrame and key column names.
        Precomputes category frequencies for the entire dataset.

        Args:
            df: DataFrame containing the data
            sample_key: Column name for sample IDs
            classification_key: Column name for category lists
        """
        self.df = df
        self.sample_key = sample_key
        self.classification_key = classification_key

        # Precompute global category statistics
        self.global_stats = self._compute_global_stats()

        # Cache for sample-specific category counts
        self.sample_cache: Dict[str, Counter] = {}

    @classmethod
    def from_collection(cls, collection: Collection, sample_key: str, classification_key: str) -> "EnrichmentAnalyzer":
        """
        Initialize the analyzer with a Collection and key column names.
        Precomputes category frequencies for the entire dataset.

        Args:
            collection: Collection containing the data
            sample_key: Column name for sample IDs
            classification_key: Column name for category lists
        """
        column_atts = [sample_key, classification_key]
        results = collection.find(select_cols=column_atts, limit=-1)
        df = results.rows_dataframe
        ea = cls(df, sample_key=sample_key, classification_key=classification_key)
        return ea

    def _compute_global_stats(self) -> Dict[str, int]:
        """
        Compute global category frequencies across all samples.
        Returns a dictionary of category -> count
        """
        global_counter = Counter()

        # Flatten all categories and count
        for categories in self.df[self.classification_key]:
            if isinstance(categories, list):
                global_counter.update(categories)
            else:
                # Handle case where categories might be a string
                global_counter.update([categories])

        return global_counter

    @property
    def sample_ids(self) -> List[str]:
        df = self.df
        return df[self.sample_key].unique().tolist()

    def _get_sample_stats(self, sample_id: str) -> Counter:
        """
        Get category frequencies for a specific sample.
        Uses caching to avoid recomputation.
        """
        if sample_id in self.sample_cache:
            return self.sample_cache[sample_id]

        sample_data = self.df[self.df[self.sample_key] == sample_id]
        if sample_data.empty:
            raise KeyError(f"Sample ID '{sample_id}' not found")
        sample_data = sample_data.dropna()
        # if sample_data.empty:
        #    raise ValueError(f"Sample ID '{sample_id}' has missing values after dropping NA")
        counter = Counter()

        for categories in sample_data[self.classification_key]:
            if isinstance(categories, list):
                counter.update(categories)
            else:
                counter.update([categories])

        self.sample_cache[sample_id] = counter
        return counter

    def find_enriched_categories(
        self,
        sample_id: str,
        min_occurrences: int = 5,
        p_value_threshold: float = 0.05,
        multiple_testing_correction: str = "bh",
    ) -> List[EnrichedCategory]:
        """
        Find categories that are enriched in the given sample.

        Args:
            sample_id: ID of the sample to analyze
            min_occurrences: Minimum number of occurrences required for a category
            p_value_threshold: P-value threshold for significance

        Returns:
            List of tuples (category, fold_change, p_value) sorted by significance
        """
        sample_stats = self._get_sample_stats(sample_id)
        total_sample_annotations = sum(sample_stats.values())
        total_global_annotations = sum(self.global_stats.values())

        results = []

        for category, sample_count in sample_stats.items():
            global_count = self.global_stats[category]

            # Skip rare categories
            if global_count < min_occurrences:
                continue

            # Calculate fold change
            sample_freq = sample_count / total_sample_annotations
            global_freq = global_count / total_global_annotations
            fold_change = sample_freq / global_freq if global_freq > 0 else float("inf")

            # Perform Fisher's exact test
            contingency_table = np.array(
                [
                    [sample_count, global_count - sample_count],
                    [
                        total_sample_annotations - sample_count,
                        total_global_annotations - total_sample_annotations - (global_count - sample_count),
                    ],
                ]
            )

            _, p_value = stats.fisher_exact(contingency_table)

            if p_value < p_value_threshold:
                results.append((category, fold_change, p_value))

        if not results:
            return results

        # Sort by p-value
        results.sort(key=lambda x: x[2])

        # Apply multiple testing correction
        categories, fold_changes, p_values = zip(*results)

        if multiple_testing_correction.lower() == "bonf":
            # Bonferroni correction
            n_tests = len(self.global_stats)  # Total number of categories tested
            adjusted_p_values = [min(1.0, p * n_tests) for p in p_values]

        elif multiple_testing_correction.lower() == "bh":
            # Benjamini-Hochberg correction
            n = len(p_values)
            sorted_indices = np.argsort(p_values)
            sorted_p_values = np.array(p_values)[sorted_indices]

            # Calculate BH adjusted p-values
            adjusted_p_values = np.zeros(n)
            for i, p in enumerate(sorted_p_values):
                adjusted_p_values[i] = p * n / (i + 1)

            # Ensure monotonicity
            for i in range(n - 2, -1, -1):
                adjusted_p_values[i] = min(adjusted_p_values[i], adjusted_p_values[i + 1])

            # Restore original order
            inverse_indices = np.argsort(sorted_indices)
            adjusted_p_values = adjusted_p_values[inverse_indices]

            # Ensure we don't exceed 1.0
            adjusted_p_values = np.minimum(adjusted_p_values, 1.0)

        else:
            # No correction
            adjusted_p_values = p_values

        # Filter by adjusted p-value threshold and create final results
        # Create EnrichedCategory objects
        final_results = [
            EnrichedCategory(category=cat, fold_change=fc, original_p_value=p, adjusted_p_value=adj_p)
            for cat, fc, p, adj_p in zip(categories, fold_changes, p_values, adjusted_p_values)
            if adj_p < p_value_threshold
        ]

        # Sort by adjusted p-value
        final_results.sort(key=lambda x: x.adjusted_p_value)
        return final_results


# Example usage:
# analyzer = EnrichmentAnalyzer(df, 'sample_id', 'categories')
# enriched = analyzer.find_enriched_categories('sample1')
# for category, fold_change, p_value in enriched:
#     print(f"{category}: {fold_change:.2f}x enrichment (p={p_value:.2e})")
