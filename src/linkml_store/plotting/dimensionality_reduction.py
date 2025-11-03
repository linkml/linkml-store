"""Dimensionality reduction utilities for embedding visualization."""

import logging
from typing import Dict, Literal, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReductionResult:
    """Container for dimensionality reduction results."""

    coordinates: np.ndarray
    method: str
    parameters: Dict
    explained_variance: Optional[float] = None

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.coordinates)

    @property
    def n_components(self) -> int:
        """Number of reduced dimensions."""
        return self.coordinates.shape[1]


def reduce_dimensions(
    vectors: np.ndarray,
    method: Literal["umap", "tsne", "pca"] = "umap",
    n_components: int = 2,
    random_state: Optional[int] = None,
    **kwargs
) -> ReductionResult:
    """
    Reduce dimensionality of embedding vectors.

    Args:
        vectors: Input vectors (n_samples, n_features)
        method: Reduction method
        n_components: Number of output dimensions
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters for the reduction method

    Returns:
        ReductionResult with reduced coordinates
    """
    if method == "pca":
        return _reduce_with_pca(vectors, n_components, random_state, **kwargs)
    elif method == "tsne":
        return _reduce_with_tsne(vectors, n_components, random_state, **kwargs)
    elif method == "umap":
        return _reduce_with_umap(vectors, n_components, random_state, **kwargs)
    else:
        raise ValueError(f"Unknown reduction method: {method}")


def _reduce_with_pca(
    vectors: np.ndarray,
    n_components: int,
    random_state: Optional[int],
    **kwargs
) -> ReductionResult:
    """Reduce dimensions using PCA."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("scikit-learn is required for PCA. Install with: pip install scikit-learn")

    pca = PCA(n_components=n_components, random_state=random_state, **kwargs)
    coordinates = pca.fit_transform(vectors)

    explained_var = sum(pca.explained_variance_ratio_) if hasattr(pca, 'explained_variance_ratio_') else None

    return ReductionResult(
        coordinates=coordinates,
        method="pca",
        parameters={"n_components": n_components, **kwargs},
        explained_variance=explained_var
    )


def _reduce_with_tsne(
    vectors: np.ndarray,
    n_components: int,
    random_state: Optional[int],
    perplexity: float = 30.0,
    learning_rate: Union[float, str] = "auto",
    n_iter: int = 1000,
    max_dimensions: int = 500,
    **kwargs
) -> ReductionResult:
    """Reduce dimensions using t-SNE."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError("scikit-learn is required for t-SNE. Install with: pip install scikit-learn")

    # Validate and debug input vectors
    logger.info(f"Input vector shape: {vectors.shape}")
    logger.info(f"Input vector dtype: {vectors.dtype}")

    # Check for NaN or Inf values
    if np.any(np.isnan(vectors)):
        nan_count = np.sum(np.isnan(vectors))
        nan_rows = np.any(np.isnan(vectors), axis=1)
        logger.warning(f"Found {nan_count} NaN values in {np.sum(nan_rows)} rows")
        # Replace NaNs with zeros as a fallback
        vectors = np.nan_to_num(vectors, nan=0.0)
        logger.info("Replaced NaN values with zeros")

    if np.any(np.isinf(vectors)):
        inf_count = np.sum(np.isinf(vectors))
        inf_rows = np.any(np.isinf(vectors), axis=1)
        logger.warning(f"Found {inf_count} Inf values in {np.sum(inf_rows)} rows")
        # Replace Infs with large finite values
        vectors = np.nan_to_num(vectors, posinf=1e10, neginf=-1e10)
        logger.info("Replaced Inf values with finite values")

    # Check vector statistics
    logger.info(f"Vector stats - min: {np.min(vectors):.6f}, max: {np.max(vectors):.6f}, mean: {np.mean(vectors):.6f}, std: {np.std(vectors):.6f}")

    # Check if all vectors are identical (can cause issues)
    if np.allclose(vectors, vectors[0]):
        logger.warning("All input vectors are identical! This will cause t-SNE to fail.")
        # Add small random noise to break symmetry
        noise = np.random.RandomState(random_state).normal(0, 1e-8, vectors.shape)
        vectors = vectors + noise
        logger.info("Added small random noise to break symmetry")

    # Check variance per dimension
    dim_variance = np.var(vectors, axis=0)
    zero_var_dims = np.sum(dim_variance == 0)
    if zero_var_dims > 0:
        logger.warning(f"Found {zero_var_dims} dimensions with zero variance")

    # t-SNE specific adjustments
    n_samples = len(vectors)
    perplexity = min(perplexity, n_samples - 1)

    # Additional perplexity validation
    if perplexity < 5:
        logger.warning(f"Perplexity {perplexity} is very low, may cause instability")
    if perplexity > n_samples / 2:
        logger.warning(f"Perplexity {perplexity} is very high relative to sample size {n_samples}")

    logger.info(f"t-SNE parameters: perplexity={perplexity}, learning_rate={learning_rate}, n_iter={n_iter}")

    # Pre-reduce with PCA if dimensions are very high (>max_dimensions)
    n_features = vectors.shape[1]
    if n_features > max_dimensions:
        logger.info(f"High dimensional data ({n_features}D). Pre-reducing with PCA to {max_dimensions}D for t-SNE stability")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(max_dimensions, n_samples - 1), random_state=random_state)
        vectors = pca.fit_transform(vectors)
        logger.info(f"PCA reduced to shape: {vectors.shape}")

    # Use max_iter instead of n_iter for newer sklearn versions
    tsne_params = {
        "n_components": n_components,
        "perplexity": perplexity,
        "learning_rate": learning_rate,
        "random_state": random_state,
        "init": "random",  # Use random init to avoid potential issues
        "method": "barnes_hut" if n_samples >= 1000 else "exact",  # Use exact for small datasets
    }

    # Handle deprecated n_iter parameter
    try:
        # Try with max_iter first (newer sklearn)
        tsne_params["max_iter"] = n_iter
        tsne = TSNE(**tsne_params, **kwargs)
    except TypeError:
        # Fall back to n_iter for older sklearn
        tsne_params["n_iter"] = n_iter
        del tsne_params["max_iter"]
        tsne = TSNE(**tsne_params, **kwargs)

    logger.info(f"Starting t-SNE fit_transform with {n_samples} samples, method: {tsne_params.get('method', 'auto')}")
    try:
        coordinates = tsne.fit_transform(vectors)
        logger.info(f"t-SNE fit transform complete")
    except Exception as e:
        logger.error(f"t-SNE failed with error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        raise

    return ReductionResult(
        coordinates=coordinates,
        method="tsne",
        parameters={
            "n_components": n_components,
            "perplexity": perplexity,
            "learning_rate": learning_rate,
            "n_iter": n_iter,
            **kwargs
        }
    )


def _reduce_with_umap(
    vectors: np.ndarray,
    n_components: int,
    random_state: Optional[int],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    **kwargs
) -> ReductionResult:
    """Reduce dimensions using UMAP."""
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")

    # UMAP specific adjustments
    n_samples = len(vectors)
    n_neighbors = min(n_neighbors, n_samples - 1)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        **kwargs
    )
    coordinates = reducer.fit_transform(vectors)

    return ReductionResult(
        coordinates=coordinates,
        method="umap",
        parameters={
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            **kwargs
        }
    )


def validate_embeddings(vectors: np.ndarray) -> Dict:
    """
    Validate and analyze embedding vectors for potential issues.

    Args:
        vectors: Input vectors to validate

    Returns:
        Dictionary with validation results and statistics
    """
    results = {
        "shape": vectors.shape,
        "dtype": str(vectors.dtype),
        "has_nan": bool(np.any(np.isnan(vectors))),
        "has_inf": bool(np.any(np.isinf(vectors))),
        "min": float(np.min(vectors)),
        "max": float(np.max(vectors)),
        "mean": float(np.mean(vectors)),
        "std": float(np.std(vectors)),
        "all_identical": bool(np.allclose(vectors, vectors[0])),
        "zero_variance_dims": int(np.sum(np.var(vectors, axis=0) == 0)),
        "n_unique_vectors": int(len(np.unique(vectors, axis=0))),
    }

    if results["has_nan"]:
        results["nan_count"] = int(np.sum(np.isnan(vectors)))
        results["nan_rows"] = int(np.sum(np.any(np.isnan(vectors), axis=1)))

    if results["has_inf"]:
        results["inf_count"] = int(np.sum(np.isinf(vectors)))
        results["inf_rows"] = int(np.sum(np.any(np.isinf(vectors), axis=1)))

    # Check for problematic patterns
    results["warnings"] = []

    if results["has_nan"]:
        results["warnings"].append(f"Contains {results.get('nan_count', 0)} NaN values")

    if results["has_inf"]:
        results["warnings"].append(f"Contains {results.get('inf_count', 0)} Inf values")

    if results["all_identical"]:
        results["warnings"].append("All vectors are identical")

    if results["zero_variance_dims"] > 0:
        results["warnings"].append(f"{results['zero_variance_dims']} dimensions have zero variance")

    if results["n_unique_vectors"] < vectors.shape[0] * 0.1:
        results["warnings"].append(f"Only {results['n_unique_vectors']} unique vectors out of {vectors.shape[0]}")

    if results["std"] < 1e-10:
        results["warnings"].append("Extremely low variance in data")

    return results


def get_optimal_parameters(
    method: str,
    n_samples: int
) -> Dict:
    """
    Get optimal parameters for dimensionality reduction based on data size.

    Args:
        method: Reduction method
        n_samples: Number of samples

    Returns:
        Dictionary of recommended parameters
    """
    if method == "pca":
        return {
            "n_components": min(2, n_samples - 1)
        }
    elif method == "tsne":
        # Adjust perplexity based on sample size
        perplexity = min(30, max(5, n_samples / 100))
        return {
            "n_components": 2,
            "perplexity": perplexity,
            "learning_rate": "auto",
            "n_iter": 1000 if n_samples < 10000 else 500
        }
    elif method == "umap":
        # Adjust n_neighbors based on sample size
        n_neighbors = min(15, max(2, int(np.sqrt(n_samples))))
        return {
            "n_components": 2,
            "n_neighbors": n_neighbors,
            "min_dist": 0.1,
            "metric": "cosine"
        }
    else:
        return {}


def cached_reduction(
    vectors: np.ndarray,
    method: str,
    cache_path: Optional[str] = None,
    force_recompute: bool = False,
    **kwargs
) -> ReductionResult:
    """
    Perform dimensionality reduction with optional caching.

    Args:
        vectors: Input vectors
        method: Reduction method
        cache_path: Path to cache file
        force_recompute: Force recomputation even if cached
        **kwargs: Additional parameters for reduction

    Returns:
        ReductionResult
    """
    import hashlib
    import pickle
    from pathlib import Path

    # Create cache key from vectors and parameters
    cache_key = None
    if cache_path and not force_recompute:
        # Create hash of input data and parameters
        hasher = hashlib.sha256()
        hasher.update(vectors.tobytes())
        hasher.update(method.encode())
        hasher.update(str(sorted(kwargs.items())).encode())
        cache_key = hasher.hexdigest()

        cache_file = Path(cache_path) / f"reduction_{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)
                logger.info(f"Loaded cached reduction from {cache_file}")
                return result
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

    # Compute reduction
    result = reduce_dimensions(vectors, method=method, **kwargs)

    # Cache result if requested
    if cache_path and cache_key:
        cache_file = Path(cache_path) / f"reduction_{cache_key}.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            logger.info(f"Cached reduction to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

    return result


def align_reductions(
    reductions: Dict[str, ReductionResult],
    reference_key: Optional[str] = None
) -> Dict[str, ReductionResult]:
    """
    Align multiple reduction results for comparison.

    Uses Procrustes analysis to align coordinates.

    Args:
        reductions: Dictionary of reduction results
        reference_key: Key of reference reduction (default: first)

    Returns:
        Dictionary of aligned reduction results
    """
    if len(reductions) <= 1:
        return reductions

    try:
        from scipy.spatial import procrustes
    except ImportError:
        logger.warning("scipy not available, returning unaligned reductions")
        return reductions

    keys = list(reductions.keys())
    if reference_key is None:
        reference_key = keys[0]

    if reference_key not in reductions:
        raise ValueError(f"Reference key {reference_key} not found")

    ref_coords = reductions[reference_key].coordinates
    aligned = {reference_key: reductions[reference_key]}

    for key in keys:
        if key == reference_key:
            continue

        # Align using Procrustes
        _, aligned_coords, _ = procrustes(ref_coords, reductions[key].coordinates)

        aligned[key] = ReductionResult(
            coordinates=aligned_coords,
            method=reductions[key].method,
            parameters=reductions[key].parameters,
            explained_variance=reductions[key].explained_variance
        )

    return aligned