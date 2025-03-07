import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


LOL = List[List[float]]


def pairwise_cosine_similarity(vector1: np.array, vector2: np.array) -> float:
    """
    Calculate the cosine similarity between two vectors.

    >>> v100 = np.array([1, 0, 0])
    >>> v010 = np.array([0, 1, 0])
    >>> v001 = np.array([0, 0, 1])
    >>> v011 = np.array([0, 1, 1])
    >>> pairwise_cosine_similarity(v100, v010)
    0.0
    >>> pairwise_cosine_similarity(v100, v001)
    0.0
    >>> pairwise_cosine_similarity(v010, v001)
    0.0
    >>> pairwise_cosine_similarity(v100, v100)
    1.0
    >>> f"{pairwise_cosine_similarity(v010, v011):0.3f}"
    '0.707'

    :param vector1:
    :param vector2:
    :return:
    """
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return float(dot_product / (norm1 * norm2))


def compute_cosine_similarity_matrix(list1: LOL, list2: LOL) -> np.ndarray:
    """
    Compute cosine similarity between two lists of vectors.

    Result is a two column vector sim[ROW][COL] where ROW is from list1 and COL is from list2.

    :param list1:
    :param list2:
    :return:
    """
    # Convert lists to numpy arrays
    matrix1 = np.array(list1)
    matrix2 = np.array(list2)

    # Normalize the vectors in both matrices
    matrix1_norm = matrix1 / np.linalg.norm(matrix1, axis=1)[:, np.newaxis]
    matrix2_norm = matrix2 / np.linalg.norm(matrix2, axis=1)[:, np.newaxis]

    # Compute dot products (resulting in cosine similarity values)
    cosine_similarity_matrix = np.dot(matrix1_norm, matrix2_norm.T)

    return cosine_similarity_matrix


def top_matches(cosine_similarity_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the top match for each row in the cosine similarity matrix.

    :param cosine_similarity_matrix:
    :return:
    """
    # Find the index of the maximum value in each row
    top_match_indices = np.argmax(cosine_similarity_matrix, axis=1)

    # Find the maximum similarity value in each row
    top_match_values = np.amax(cosine_similarity_matrix, axis=1)

    return top_match_indices, top_match_values


def top_n_matches(cosine_similarity_matrix: np.ndarray, n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    # Find the indices that would sort each row in descending order
    sorted_indices = np.argsort(-cosine_similarity_matrix, axis=1)

    # Take the first n indices from the sorted indices to get the top n matches
    top_n_indices = sorted_indices[:, :n]

    # Take the first n values from the sorted values to get the top n match values
    top_n_values = -np.sort(-cosine_similarity_matrix, axis=1)[:, :n]

    return top_n_indices, top_n_values


def mmr_diversified_search(
    query_vector: np.ndarray, document_vectors: List[np.ndarray], relevance_factor=0.5, top_n=None
) -> List[int]:
    """
    Perform diversified search using Maximal Marginal Relevance (MMR).

    :param query_vector: The vector representing the query.
    :param document_vectors: The vectors representing the documents.
    :param relevance_factor: The balance parameter between relevance and diversity.
    :param top_n: The number of results to return. If None, return all.
    :return: A list of indices representing the diversified order of documents.
    """
    if top_n is None:
        # If no specific number of results is specified, return all
        top_n = len(document_vectors)

    if top_n == 0:
        return []

    # Calculate cosine similarities between query and all documents
    norms_query = np.linalg.norm(query_vector)
    norms_docs = np.linalg.norm(document_vectors, axis=1)
    similarities = np.dot(document_vectors, query_vector) / (norms_docs * norms_query)

    # Initialize set of selected indices and results list
    selected_indices = set()
    result_indices = []

    # Diversified search loop
    for _ in range(top_n):
        max_mmr = float("-inf")
        best_index = None

        # Loop over all documents
        for idx, _doc_vector in enumerate(document_vectors):
            if idx not in selected_indices:
                relevance = relevance_factor * similarities[idx]
                diversity = 0

                # Penalize based on similarity to already selected documents
                if selected_indices:
                    max_sim_to_selected = max(
                        [
                            np.dot(document_vectors[idx], document_vectors[s])
                            / (np.linalg.norm(document_vectors[idx]) * np.linalg.norm(document_vectors[s]))
                            for s in selected_indices
                        ]
                    )
                    diversity = (1 - relevance_factor) * max_sim_to_selected

                mmr_score = relevance - diversity

                # Update best MMR score and index
                if mmr_score > max_mmr:
                    max_mmr = mmr_score
                    best_index = idx

        # Add the best document to the result and mark it as selected
        if best_index is None:
            logger.warning(f"No best index found over {len(document_vectors)} documents.")
            continue
        result_indices.append(best_index)
        selected_indices.add(best_index)

    return result_indices
