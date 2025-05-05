import os

import numpy as np
import pytest

from linkml_store.index import get_indexer
from tests import INPUT_DIR

INDEX_CLASSES = ["simple", "llm"]

LLM_CACHE = INPUT_DIR / "llm_cache.db"


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize(
    "texts",
    [
        {
            "a": "hello world",
            "b": "goodbye world",
            "c": "hello goodbye",
            "d": "goodbye world",
            "e": "goodbye universe",
            "aa": "hello world, hello world",
            "bb": "goodbye world, goodbye world",
            "ee": "goodbye universe, goodbye universe",
            "f": "!@#$%^&*()",
            "a100": "a" * 100,
            "a1000": "a" * 1000,
            "ab100": "ab" * 100,
            "ab1000": "ab" * 1000,
        },
    ],
)
def test_index(index_class, texts):
    """Test indexing functionality, skipping tests if required API key is missing."""

    # Skip LLMIndex tests in GitHub Actions
    if os.environ.get("GITHUB_ACTIONS") == "true" and index_class == "llm":
        pytest.skip("Skipping LLMIndex test in GitHub Actions")

    # Check for required API key
    required_api_key = os.getenv("LLM_API_KEY")
    if index_class == "llm" and not required_api_key:
        pytest.skip("Skipping LLMIndex test: API key not found.")

    # Proceed with test
    index = get_indexer(index_class)
    if index_class == "llm":
        index.cached_embeddings_database = str(LLM_CACHE)
        index.cache_queries = True

    vectors = index.texts_to_vectors(texts.values())
    id_vector_tuples = list(zip(texts.keys(), vectors))

    for text_id, text in texts.items():
        results = index.search(text, id_vector_tuples)
        # Ensure the queried text appears at the top of the search results
        exact_matches = [r[1] for r in results if np.isclose(r[0], 1.0, rtol=1e-3)]
        assert text_id in exact_matches, f"Exact match not found in : {results}"
