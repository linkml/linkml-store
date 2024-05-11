import hashlib
import logging

import numpy as np

from linkml_store.index.indexer import INDEX_ITEM, Indexer

logger = logging.getLogger(__name__)


class SimpleIndexer(Indexer):
    """
    A implementations index that uses a hash function to generate an index from text.

    This uses a naive method to generate an index from text. It is not suitable for production use.
    """

    def text_to_vector(self, text: str, cache: bool = None, **kwargs) -> INDEX_ITEM:
        """
        This is a naive method purely for testing

        :param text:
        :return:
        """
        vector_length = self.vector_default_length
        text = text.lower()
        # trigrams
        words = [text[i : i + 3] for i in range(len(text) - 2)]

        vector = np.zeros(vector_length, dtype=float)

        # Iterate over each trigram in the text
        for word in words:
            # Generate a hash value for the word
            hash_value = int(hashlib.sha1(word.encode("utf-8")).hexdigest(), 16)

            # Compute the index in the vector using modulo
            index = hash_value % vector_length

            # Increment the count at the computed index
            vector[index] += 1.0
        logger.debug(f"Indexed text: {text} as {vector}")
        return vector
