import os

import numpy as np
import pytest

from linkml_store.index.implementations.llm_index import LLMIndex
from linkml_store.index.implementations.simple_index import SimpleIndex

INDEX_CLASSES = [SimpleIndex, LLMIndex]


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("texts", [
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
            #"z": "",
        },
])
def test_index(index_class, texts):
    if os.environ.get('GITHUB_ACTIONS') == 'true' and index_class == LLMIndex:
        pytest.skip("Skipping LLMIndex test in GitHub Actions")
    index = index_class()
    vectors = index.texts_to_vectors(texts.values())
    id_vector_tuples = list(zip(texts.keys(), vectors))
    for text_id, text in texts.items():
        results = index.search(text, id_vector_tuples)
        # check that the text_id is in the results and is the top or joint top result
        exact_matches = [r[1] for r in results if np.isclose(r[0], 1.0, rtol=1e-3)]
        assert text_id in exact_matches, f"Exact match not found in : {results}"

