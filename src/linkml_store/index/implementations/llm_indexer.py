import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from linkml_store.api.config import CollectionConfig
from linkml_store.index.indexer import INDEX_ITEM, Indexer
from linkml_store.utils.llm_utils import get_token_limit, render_formatted_text

if TYPE_CHECKING:
    import llm

CHUNK_SIZE = 1000

logger = logging.getLogger(__name__)


class LLMIndexer(Indexer):
    """
    An indexer that wraps the llm library.

    This indexer is used to convert text to vectors using the llm library.

    >>> indexer = LLMIndexer(cached_embeddings_database="tests/input/llm_cache.db")
    >>> vector = indexer.text_to_vector("hello")

    TODO: Implement true batching for embedding API calls
    TODO: Add batch_size parameter to control batch processing
    TODO: Support batch embedding APIs (e.g., OpenAI batch endpoint)
    TODO: Add progress reporting for large batch operations
    TODO: Implement smart batching with accumulation and flushing
    """

    embedding_model_name: str = "text-embedding-ada-002"
    _embedding_model: "llm.EmbeddingModel" = None
    cached_embeddings_database: str = None
    cached_embeddings_collection: str = None
    cache_queries: bool = False
    truncation_method: Optional[str] = None
    # TODO: Add batch_size: int = 100 parameter for batch processing
    # TODO: Add supported_models class variable with model metadata (dims, costs, limits)
    # TODO: Add model_validation to check if model exists before use

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

    def texts_to_vectors(
        self, texts: List[str], cache: bool = None, token_limit_penalty=0, batch_size: int=None, **kwargs
    ) -> List[INDEX_ITEM]:
        """
        Use LLM to embed.

        >>> indexer = LLMIndexer(cached_embeddings_database="tests/input/llm_cache.db")
        >>> vectors = indexer.texts_to_vectors(["hello", "goodbye"])

        :param texts:
        :param cache:
        :param token_limit_penalty:
        :return:
        """
        from tiktoken import encoding_for_model

        logging.info(f"Converting {len(texts)} texts to vectors")
        model = self.embedding_model
        # TODO: make this more accurate
        token_limit = get_token_limit(model.model_id) - token_limit_penalty
        logging.info(f"Token limit for {model.model_id}: {token_limit}")
        encoding = encoding_for_model(self.embedding_model_name)

        def truncate_text(text: str) -> str:
            # split into tokens every 1000 chars:
            parts = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
            truncated = render_formatted_text(
                lambda x: "".join(x),
                parts,
                encoding,
                token_limit,
            )
            logger.debug(f"Truncated text from {len(text)} to {len(truncated)}")
            return truncated

        texts = [truncate_text(text) for text in texts]
        # Calculate average number of tokens per text for accurate batch sizing
        text_token_counts = [len(encoding.encode(t)) for t in texts]
        avg_text_tokens = sum(text_token_counts) / len(text_token_counts)
        logger.info(f"Average text token count: {avg_text_tokens}")
        if batch_size is None:
            # TODO: empirically determine best batch size
            batch_size = max(int(token_limit / avg_text_tokens), 5)
            logger.info(f"Setting batch size to {batch_size}")
        

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
                alias=coll_name,
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

            embeddings = list([None] * len(texts))
            uncached_texts = []
            n = 0
            # TODO: Implement batch lookup for cache checking (single query for all texts)
            # TODO: Use IN clause or batch query to check multiple texts at once
            logger.info(f"Checking cache for {len(texts)} texts")
            for i in range(len(texts)):
                # TODO: optimize this - currently makes N database queries for N texts
                text = texts[i]
                logger.debug(f"Looking for cached embedding for {text}")
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
                uncached_embeddings = list(model.embed_multi(uncached_texts, batch_size=batch_size))
                # TODO: Combine into a single insert with multiple rows for better performance
                # TODO: Use insert_many or bulk insert instead of individual inserts
                for i, index in enumerate(uncached_indices):
                    logger.debug(f"Indexing text at {i}")
                    embeddings[index] = uncached_embeddings[i]
                    embeddings_collection.insert(
                        {"text": uncached_texts[i], "embedding": embeddings[index], "model_id": model_id}
                    )
                embeddings_collection.commit()
        else:
            logger.info(f"Embedding {len(texts)} texts")
            # TODO: Add progress callback for large batches without cache
            embeddings = list(model.embed_multi(texts, batch_size=batch_size))
        return [np.array(v, dtype=float) for v in embeddings]
