import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np

from linkml_store.api.config import CollectionConfig
from linkml_store.index.indexer import INDEX_ITEM, Indexer

if TYPE_CHECKING:
    import llm


logger = logging.getLogger(__name__)


class LLMIndexer(Indexer):
    """
    An indexer that wraps the llm library.

    This indexer is used to convert text to vectors using the llm library.

    >>> indexer = LLMIndexer(cached_embeddings_database="tests/input/llm_cache.db")
    >>> vector = indexer.text_to_vector("hello")
    """

    embedding_model_name: str = "ada-002"
    _embedding_model: "llm.EmbeddingModel" = None
    cached_embeddings_database: str = None
    cached_embeddings_collection: str = None
    cache_queries: bool = False

    @property
    def embedding_model(self):
        import llm

        if self._embedding_model is None:
            self._embedding_model = llm.get_embedding_model(self.embedding_model_name)
        return self._embedding_model

    def text_to_vector(self, text: str, cache: bool = None, **kwargs) -> INDEX_ITEM:
        """
        Convert a text to an indexable object

        >>> indexer = LLMIndexer(cached_embeddings_database="tests/input/llm_cache.db")
        >>> vector = indexer.text_to_vector("hello")

        :param text:
        :return:
        """
        return self.texts_to_vectors([text], cache=cache, **kwargs)[0]

    def texts_to_vectors(self, texts: List[str], cache: bool = None, **kwargs) -> List[INDEX_ITEM]:
        """
        Use LLM to embed

        >>> indexer = LLMIndexer(cached_embeddings_database="tests/input/llm_cache.db")
        >>> vectors = indexer.texts_to_vectors(["hello", "goodbye"])

        :param texts:
        :return:
        """
        logging.info(f"Converting {len(texts)} texts to vectors")
        model = self.embedding_model
        if self.cached_embeddings_database and (cache is None or cache or self.cache_queries):
            model_id = model.model_id
            if not model_id:
                raise ValueError("Model ID is required to cache embeddings")
            db_path = Path(self.cached_embeddings_database)
            coll_name = self.cached_embeddings_collection
            if not coll_name:
                coll_name = "all_embeddings"
            from linkml_store import Client

            embeddings_client = Client()
            config = CollectionConfig(
                name=coll_name,
                type="Embeddings",
                attributes={
                    "text": {"range": "string"},
                    "model_id": {"range": "string"},
                    "embedding": {"range": "float", "array": {}},
                },
            )
            embeddings_db = embeddings_client.get_database(f"duckdb:///{db_path}")
            if coll_name in embeddings_db.list_collection_names():
                # Load existing collection and use its model
                embeddings_collection = embeddings_db.create_collection(coll_name, metadata=config)
            else:
                embeddings_collection = embeddings_db.create_collection(coll_name, metadata=config)
            texts = list(texts)
            embeddings = list([None] * len(texts))
            uncached_texts = []
            n = 0
            for i in range(len(texts)):
                # TODO: optimize this
                text = texts[i]
                logger.info(f"Looking for cached embedding for {text}")
                r = embeddings_collection.find({"text": text, "model_id": model_id})
                if r.num_rows:
                    embeddings[i] = r.rows[0]["embedding"]
                    n += 1
                    logger.info("Found")
                else:
                    uncached_texts.append((text, i))
                    logger.info("NOT Found")
            logger.info(f"Found {n} cached embeddings")
            if uncached_texts:
                logger.info(f"Embedding {len(uncached_texts)} uncached texts")
                uncached_texts, uncached_indices = zip(*uncached_texts)
                uncached_embeddings = list(model.embed_multi(uncached_texts))
                # TODO: combine into a single insert with multiple rows
                for i, index in enumerate(uncached_indices):
                    logger.debug(f"Indexing text at {i}")
                    embeddings[index] = uncached_embeddings[i]
                    embeddings_collection.insert(
                        {"text": uncached_texts[i], "embedding": embeddings[index], "model_id": model_id}
                    )
        else:
            logger.info(f"Embedding {len(texts)} texts")
            embeddings = model.embed_multi(texts)
        return [np.array(v, dtype=float) for v in embeddings]
