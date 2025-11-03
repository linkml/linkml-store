"""Utilities for finding matches between embeddings in collections."""

import logging
from typing import Dict, List, Optional, Tuple, Literal, Any
from dataclasses import dataclass, field
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class DistanceMetric(str, Enum):
    """Distance metrics for similarity computation."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    L2 = "l2"  # Alias for euclidean
    DOT = "dot"
    MANHATTAN = "manhattan"


@dataclass
class MatchResult:
    """Result of a single match between items."""
    source_id: str
    source_data: Dict[str, Any]
    target_id: str
    target_data: Dict[str, Any]
    similarity: float
    distance: float
    rank: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "similarity": self.similarity,
            "distance": self.distance,
            "rank": self.rank,
            "source": self.source_data,
            "target": self.target_data
        }


@dataclass
class MatchingConfig:
    """Configuration for matching operations."""
    metric: DistanceMetric = DistanceMetric.COSINE
    max_matches_per_item: int = 5
    similarity_threshold: Optional[float] = None
    distance_threshold: Optional[float] = None
    source_fields: Optional[List[str]] = None
    target_fields: Optional[List[str]] = None
    exclude_self_matches: bool = True
    normalize_vectors: bool = False
    batch_size: int = 100


@dataclass
class MatchingResults:
    """Container for all matching results."""
    matches: List[MatchResult]
    config: MatchingConfig
    source_collection: str
    target_collection: str
    total_source_items: int
    total_target_items: int

    @property
    def num_matches(self) -> int:
        """Total number of matches found."""
        return len(self.matches)

    def get_matches_for_source(self, source_id: str) -> List[MatchResult]:
        """Get all matches for a specific source item."""
        return [m for m in self.matches if m.source_id == source_id]

    def to_dataframe(self):
        """Convert to pandas DataFrame for analysis."""
        try:
            import pandas as pd
            data = []
            for match in self.matches:
                row = {
                    "source_id": match.source_id,
                    "target_id": match.target_id,
                    "similarity": match.similarity,
                    "distance": match.distance,
                    "rank": match.rank
                }
                # Add source fields
                for key, value in match.source_data.items():
                    row[f"source_{key}"] = value
                # Add target fields
                for key, value in match.target_data.items():
                    row[f"target_{key}"] = value
                data.append(row)
            return pd.DataFrame(data)
        except ImportError:
            logger.warning("pandas not installed, cannot convert to DataFrame")
            return None


def compute_similarity_matrix(
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
    metric: DistanceMetric = DistanceMetric.COSINE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute similarity and distance matrices between source and target vectors.

    Args:
        source_vectors: Source embeddings (n_source, n_dims)
        target_vectors: Target embeddings (n_target, n_dims)
        metric: Distance metric to use

    Returns:
        Tuple of (similarity_matrix, distance_matrix)
    """
    if metric in [DistanceMetric.COSINE, DistanceMetric.DOT]:
        # Normalize for cosine similarity
        if metric == DistanceMetric.COSINE:
            source_norm = source_vectors / (np.linalg.norm(source_vectors, axis=1, keepdims=True) + 1e-10)
            target_norm = target_vectors / (np.linalg.norm(target_vectors, axis=1, keepdims=True) + 1e-10)
        else:
            source_norm = source_vectors
            target_norm = target_vectors

        # Compute similarity matrix
        similarity_matrix = np.dot(source_norm, target_norm.T)
        distance_matrix = 1 - similarity_matrix

    elif metric in [DistanceMetric.EUCLIDEAN, DistanceMetric.L2]:
        # Compute pairwise Euclidean distances
        # Using broadcasting: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        source_sq = np.sum(source_vectors**2, axis=1, keepdims=True)
        target_sq = np.sum(target_vectors**2, axis=1, keepdims=True)
        dot_product = np.dot(source_vectors, target_vectors.T)

        distance_matrix = np.sqrt(np.maximum(0, source_sq + target_sq.T - 2 * dot_product))
        # Convert distance to similarity (inverse relationship)
        max_dist = np.max(distance_matrix)
        if max_dist > 0:
            similarity_matrix = 1 - (distance_matrix / max_dist)
        else:
            similarity_matrix = 1 - distance_matrix

    elif metric == DistanceMetric.MANHATTAN:
        # Manhattan distance (L1)
        distance_matrix = np.zeros((len(source_vectors), len(target_vectors)))
        for i, source_vec in enumerate(source_vectors):
            for j, target_vec in enumerate(target_vectors):
                distance_matrix[i, j] = np.sum(np.abs(source_vec - target_vec))

        max_dist = np.max(distance_matrix)
        if max_dist > 0:
            similarity_matrix = 1 - (distance_matrix / max_dist)
        else:
            similarity_matrix = 1 - distance_matrix
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return similarity_matrix, distance_matrix


def find_best_matches(
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
    source_ids: List[str],
    target_ids: List[str],
    source_metadata: List[Dict],
    target_metadata: List[Dict],
    config: MatchingConfig
) -> List[MatchResult]:
    """
    Find best matches between source and target embeddings.

    Args:
        source_vectors: Source embedding vectors
        target_vectors: Target embedding vectors
        source_ids: IDs for source items
        target_ids: IDs for target items
        source_metadata: Metadata for source items
        target_metadata: Metadata for target items
        config: Matching configuration

    Returns:
        List of match results
    """
    # Normalize vectors if requested
    if config.normalize_vectors:
        source_vectors = source_vectors / (np.linalg.norm(source_vectors, axis=1, keepdims=True) + 1e-10)
        target_vectors = target_vectors / (np.linalg.norm(target_vectors, axis=1, keepdims=True) + 1e-10)

    # Compute similarity and distance matrices
    similarity_matrix, distance_matrix = compute_similarity_matrix(
        source_vectors, target_vectors, config.metric
    )

    matches = []

    # Find best matches for each source item
    for i, source_id in enumerate(source_ids):
        # Get similarities and distances for this source item
        similarities = similarity_matrix[i]
        distances = distance_matrix[i]

        # Create pairs of (index, similarity, distance)
        candidates = []
        for j, (sim, dist) in enumerate(zip(similarities, distances)):
            target_id = target_ids[j]

            # Skip self-matches if configured
            if config.exclude_self_matches and source_id == target_id:
                continue

            # Apply thresholds
            if config.similarity_threshold is not None and sim < config.similarity_threshold:
                continue
            if config.distance_threshold is not None and dist > config.distance_threshold:
                continue

            candidates.append((j, sim, dist))

        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Take top N matches
        for rank, (j, sim, dist) in enumerate(candidates[:config.max_matches_per_item], 1):
            # Extract metadata fields
            source_data = {}
            if config.source_fields:
                source_data = {k: source_metadata[i].get(k) for k in config.source_fields}
            else:
                source_data = source_metadata[i]

            target_data = {}
            if config.target_fields:
                target_data = {k: target_metadata[j].get(k) for k in config.target_fields}
            else:
                target_data = target_metadata[j]

            match = MatchResult(
                source_id=source_id,
                source_data=source_data,
                target_id=target_ids[j],
                target_data=target_data,
                similarity=float(sim),
                distance=float(dist),
                rank=rank
            )
            matches.append(match)

    return matches


def match_embeddings_between_collections(
    database,
    source_collection: str,
    target_collection: str,
    index_name: str = None,
    config: Optional[MatchingConfig] = None,
    limit: Optional[int] = None
) -> MatchingResults:
    """
    Find matches between embeddings in two collections.

    Args:
        database: LinkML database object
        source_collection: Name of source collection
        target_collection: Name of target collection
        index_name: Name of index to use (defaults to first available)
        config: Matching configuration
        limit: Limit number of items to process

    Returns:
        MatchingResults object
    """
    if config is None:
        config = MatchingConfig()

    from linkml_store.utils.embedding_utils import extract_embeddings_from_collection

    # Extract embeddings from source collection
    logger.info(f"Extracting embeddings from source collection: {source_collection}")
    source_coll = database.get_collection(source_collection)
    source_data = extract_embeddings_from_collection(
        source_coll,
        index_name=index_name,
        limit=limit,
        include_metadata=True
    )

    # Extract embeddings from target collection
    logger.info(f"Extracting embeddings from target collection: {target_collection}")
    target_coll = database.get_collection(target_collection)
    target_data = extract_embeddings_from_collection(
        target_coll,
        index_name=index_name,
        limit=limit,
        include_metadata=True
    )

    # Find matches
    logger.info(f"Finding matches between {source_data.n_samples} source and {target_data.n_samples} target items")
    matches = find_best_matches(
        source_vectors=source_data.vectors,
        target_vectors=target_data.vectors,
        source_ids=source_data.object_ids,
        target_ids=target_data.object_ids,
        source_metadata=source_data.metadata,
        target_metadata=target_data.metadata,
        config=config
    )

    return MatchingResults(
        matches=matches,
        config=config,
        source_collection=source_collection,
        target_collection=target_collection,
        total_source_items=source_data.n_samples,
        total_target_items=target_data.n_samples
    )


def match_embeddings_within_collection(
    database,
    collection_name: str,
    index_name: str = None,
    config: Optional[MatchingConfig] = None,
    limit: Optional[int] = None
) -> MatchingResults:
    """
    Find matches within a single collection (self-similarity).

    Args:
        database: LinkML database object
        collection_name: Name of collection
        index_name: Name of index to use
        config: Matching configuration
        limit: Limit number of items

    Returns:
        MatchingResults object
    """
    if config is None:
        config = MatchingConfig()

    # Ensure self-matches are excluded for within-collection matching
    config.exclude_self_matches = True

    # Use same collection as both source and target
    return match_embeddings_between_collections(
        database=database,
        source_collection=collection_name,
        target_collection=collection_name,
        index_name=index_name,
        config=config,
        limit=limit
    )


def format_matches_report(
    results: MatchingResults,
    max_examples: int = 10
) -> str:
    """
    Format matching results as a human-readable report.

    Args:
        results: Matching results
        max_examples: Maximum examples to show

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("EMBEDDING MATCHING REPORT")
    lines.append("=" * 60)

    lines.append(f"\nSource Collection: {results.source_collection}")
    lines.append(f"Target Collection: {results.target_collection}")
    lines.append(f"Source Items: {results.total_source_items}")
    lines.append(f"Target Items: {results.total_target_items}")
    lines.append(f"Total Matches: {results.num_matches}")

    lines.append(f"\nConfiguration:")
    lines.append(f"  Metric: {results.config.metric.value}")
    lines.append(f"  Max matches per item: {results.config.max_matches_per_item}")
    if results.config.similarity_threshold:
        lines.append(f"  Similarity threshold: {results.config.similarity_threshold}")
    if results.config.distance_threshold:
        lines.append(f"  Distance threshold: {results.config.distance_threshold}")

    # Show top matches
    if results.matches:
        lines.append(f"\nTop {min(max_examples, len(results.matches))} Matches:")
        lines.append("-" * 60)

        # Sort by similarity for display
        sorted_matches = sorted(results.matches, key=lambda m: m.similarity, reverse=True)

        for i, match in enumerate(sorted_matches[:max_examples], 1):
            lines.append(f"\n{i}. Similarity: {match.similarity:.4f} | Distance: {match.distance:.4f}")
            lines.append(f"   Source [{match.source_id}]:")
            for key, value in match.source_data.items():
                if value:
                    lines.append(f"     {key}: {str(value)[:100]}")
            lines.append(f"   Target [{match.target_id}]:")
            for key, value in match.target_data.items():
                if value:
                    lines.append(f"     {key}: {str(value)[:100]}")

    # Summary statistics
    if results.matches:
        similarities = [m.similarity for m in results.matches]
        lines.append("\nSummary Statistics:")
        lines.append(f"  Mean similarity: {np.mean(similarities):.4f}")
        lines.append(f"  Std similarity: {np.std(similarities):.4f}")
        lines.append(f"  Min similarity: {np.min(similarities):.4f}")
        lines.append(f"  Max similarity: {np.max(similarities):.4f}")

    return "\n".join(lines)