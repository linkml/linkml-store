"""Utilities for extracting and processing embeddings from indexed collections."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingData:
    """Container for embedding data from collections."""

    vectors: np.ndarray
    metadata: List[Dict]
    collection_names: List[str]
    collection_indices: List[int]
    object_ids: List[str]

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.vectors)

    @property
    def n_dimensions(self) -> int:
        """Number of dimensions in embeddings."""
        return self.vectors.shape[1] if len(self.vectors.shape) > 1 else 0

    def get_metadata_values(self, field: str) -> List:
        """Extract values for a specific metadata field."""
        return [m.get(field) for m in self.metadata]


def extract_embeddings_from_collection(
    collection,
    index_name: str = None,
    limit: Optional[int] = None,
    include_metadata: bool = True,
    metadata_fields: Optional[List[str]] = None
) -> EmbeddingData:
    """
    Extract embeddings from an indexed collection.

    Args:
        collection: LinkML collection object
        index_name: Name of the index to use (defaults to first available)
        limit: Maximum number of embeddings to extract
        include_metadata: Whether to include source object metadata
        metadata_fields: Specific metadata fields to include (None = all)

    Returns:
        EmbeddingData object containing vectors and metadata
    """
    # Get the index name - handle collections without loaded indexers
    if index_name is None:
        # Try to find index collections directly
        db = collection.parent
        all_collections = db.list_collection_names()
        # TODO: use the indexer metadata to find the index name
        index_prefix = f"internal__index__{collection.alias}__"
        index_collections = [c for c in all_collections if c.startswith(index_prefix)]

        if not index_collections:
            raise ValueError(f"Collection {collection.alias} has no indexes")

        # Extract index name from first index collection
        index_name = index_collections[0].replace(index_prefix, "")
        if len(index_collections) > 1:
            logger.warning(f"Multiple indexes found, using: {index_name}")

    # Get the index collection
    index_collection_name = f"internal__index__{collection.alias}__{index_name}"
    index_collection = collection.parent.get_collection(index_collection_name)

    # Query the index collection
    query_result = index_collection.find(limit=limit)

    if query_result.num_rows == 0:
        raise ValueError(f"No indexed data found in {index_collection_name}")

    vectors = []
    metadata = []
    object_ids = []

    for row in query_result.rows:
        # Extract vector (usually stored in __index__ field)
        vector = row.get("__index__")
        if vector is None:
            logger.warning(f"No vector found for object {row.get('id')}")
            continue

        vectors.append(vector)

        # Extract object ID
        obj_id = row.get("id") or row.get("_id") or str(len(vectors))
        object_ids.append(obj_id)

        # Extract metadata
        if include_metadata:
            meta = {}
            if metadata_fields:
                # Only include specified fields
                for field in metadata_fields:
                    if field in row:
                        meta[field] = row[field]
            else:
                # Include all fields except the vector
                meta = {k: v for k, v in row.items() if k != "__index__"}
            metadata.append(meta)

    return EmbeddingData(
        vectors=np.array(vectors),
        metadata=metadata,
        collection_names=[collection.alias] * len(vectors),
        collection_indices=[0] * len(vectors),
        object_ids=object_ids
    )


def extract_embeddings_from_multiple_collections(
    database,
    collection_names: List[str],
    index_name: Optional[str] = None,
    limit_per_collection: Optional[int] = None,
    include_metadata: bool = True,
    metadata_fields: Optional[List[str]] = None,
    normalize: bool = False
) -> EmbeddingData:
    """
    Extract embeddings from multiple collections.

    Args:
        database: LinkML database object
        collection_names: List of collection names
        index_name: Name of index to use (must be same across collections)
        limit_per_collection: Max embeddings per collection
        include_metadata: Whether to include source object metadata
        metadata_fields: Specific metadata fields to include
        normalize: Whether to normalize vectors to unit length

    Returns:
        Combined EmbeddingData object
    """
    all_vectors = []
    all_metadata = []
    all_collection_names = []
    all_collection_indices = []
    all_object_ids = []

    for i, coll_name in enumerate(collection_names):
        try:
            collection = database.get_collection(coll_name)
            data = extract_embeddings_from_collection(
                collection,
                index_name=index_name,
                limit=limit_per_collection,
                include_metadata=include_metadata,
                metadata_fields=metadata_fields
            )

            all_vectors.append(data.vectors)
            all_metadata.extend(data.metadata)
            all_collection_names.extend([coll_name] * data.n_samples)
            all_collection_indices.extend([i] * data.n_samples)
            all_object_ids.extend(data.object_ids)

        except Exception as e:
            logger.error(f"Failed to extract embeddings from {coll_name}: {e}")
            continue

    if not all_vectors:
        raise ValueError("No embeddings extracted from any collection")

    combined_vectors = np.vstack(all_vectors)

    # Normalize if requested
    if normalize:
        # Ensure vectors are float type for division
        combined_vectors = combined_vectors.astype(np.float64)
        norms = np.linalg.norm(combined_vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        combined_vectors = combined_vectors / norms

    return EmbeddingData(
        vectors=combined_vectors,
        metadata=all_metadata,
        collection_names=all_collection_names,
        collection_indices=all_collection_indices,
        object_ids=all_object_ids
    )


def sample_embeddings(
    embedding_data: EmbeddingData,
    n_samples: int = 1000,
    method: str = "random",
    random_state: Optional[int] = None
) -> EmbeddingData:
    """
    Sample embeddings for visualization.

    Args:
        embedding_data: Original embedding data
        n_samples: Number of samples to select
        method: Sampling method ('random', 'uniform', 'density')
        random_state: Random seed for reproducibility

    Returns:
        Sampled EmbeddingData object
    """
    if embedding_data.n_samples <= n_samples:
        return embedding_data

    if random_state is not None:
        np.random.seed(random_state)

    if method == "random":
        indices = np.random.choice(
            embedding_data.n_samples,
            size=n_samples,
            replace=False
        )
    elif method == "uniform":
        # Sample uniformly across collections
        indices = []
        for coll_idx in set(embedding_data.collection_indices):
            coll_mask = np.array(embedding_data.collection_indices) == coll_idx
            coll_indices = np.where(coll_mask)[0]
            n_from_coll = min(
                len(coll_indices),
                n_samples // len(set(embedding_data.collection_indices))
            )
            indices.extend(
                np.random.choice(coll_indices, size=n_from_coll, replace=False)
            )
        indices = np.array(indices[:n_samples])
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return EmbeddingData(
        vectors=embedding_data.vectors[indices],
        metadata=[embedding_data.metadata[i] for i in indices],
        collection_names=[embedding_data.collection_names[i] for i in indices],
        collection_indices=[embedding_data.collection_indices[i] for i in indices],
        object_ids=[embedding_data.object_ids[i] for i in indices]
    )


def compute_embedding_statistics(embedding_data: EmbeddingData) -> Dict:
    """
    Compute statistics about embeddings.

    Args:
        embedding_data: Embedding data

    Returns:
        Dictionary of statistics
    """
    stats = {
        "n_samples": embedding_data.n_samples,
        "n_dimensions": embedding_data.n_dimensions,
        "n_collections": len(set(embedding_data.collection_names)),
        "collections": list(set(embedding_data.collection_names)),
    }

    # Per-collection counts
    from collections import Counter
    collection_counts = Counter(embedding_data.collection_names)
    stats["samples_per_collection"] = dict(collection_counts)

    # Vector statistics
    if embedding_data.n_samples > 0:
        stats["mean_norm"] = float(np.mean(np.linalg.norm(embedding_data.vectors, axis=1)))
        stats["std_norm"] = float(np.std(np.linalg.norm(embedding_data.vectors, axis=1)))

        # Compute average pairwise similarity (on sample if large)
        sample_size = min(100, embedding_data.n_samples)
        if sample_size > 1:
            sample_indices = np.random.choice(
                embedding_data.n_samples,
                size=sample_size,
                replace=False
            )
            sample_vectors = embedding_data.vectors[sample_indices]

            # Normalize for cosine similarity
            norms = np.linalg.norm(sample_vectors, axis=1, keepdims=True)
            normalized = sample_vectors / (norms + 1e-10)
            similarities = np.dot(normalized, normalized.T)

            # Extract upper triangle (excluding diagonal)
            upper_tri = similarities[np.triu_indices(sample_size, k=1)]
            stats["mean_similarity"] = float(np.mean(upper_tri))
            stats["std_similarity"] = float(np.std(upper_tri))

    return stats