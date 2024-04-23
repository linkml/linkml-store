from typing import TYPE_CHECKING, List

import numpy as np

from linkml_store.index.indexer import INDEX_ITEM, Indexer

if TYPE_CHECKING:
    import llm


class LLMIndexer(Indexer):
    """
    A implementations index wraps the llm library
    """

    embedding_model_name: str = "ada-002"
    _embedding_model: "llm.EmbeddingModel" = None

    @property
    def embedding_model(self):
        import llm

        if self._embedding_model is None:
            self._embedding_model = llm.get_embedding_model(self.embedding_model_name)
        return self._embedding_model

    def text_to_vector(self, text: str) -> INDEX_ITEM:
        """
        Convert a text to an indexable object

        :param text:
        :return:
        """
        return self.texts_to_vectors([text])[0]

    def texts_to_vectors(self, texts: List[str]) -> List[INDEX_ITEM]:
        """
        Use LLM to embed

        :param texts:
        :return:
        """
        embeddings = self.embedding_model.embed_multi(texts)
        return [np.array(v, dtype=float) for v in embeddings]
