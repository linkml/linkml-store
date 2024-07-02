"""
Indexers package.

Indexers allow indexes to be added to existing :class:`Collection` objects.

Current two are supported:

* simple: :class:`SimpleIndexer`
* llm: :class:`LLMIndexer`
"""

from typing import Type

from linkml_store.index.implementations.llm_indexer import LLMIndexer
from linkml_store.index.implementations.simple_indexer import SimpleIndexer
from linkml_store.index.indexer import Indexer

INDEXER_CLASSES = {
    "simple": SimpleIndexer,
    "llm": LLMIndexer,
}


def get_indexer_class(name: str) -> Type[Indexer]:
    """
    Get an indexer class by name.

    :param name: the name of the indexer (simple, llm, ...)
    :return: the indexer class
    """
    if name not in INDEXER_CLASSES:
        raise ValueError(f"Unknown indexer class: {name}")
    return INDEXER_CLASSES[name]


def get_indexer(index_type: str, **kwargs) -> Indexer:
    """
    Get an indexer by name.

    >>> simple_indexer = get_indexer("simple")
    >>> llm_indexer = get_indexer("llm")

    :param name: the name of the indexer (simple, llm, ...)
    :param kwargs: additional arguments to pass to the indexer
    :return: the indexer
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    cls = get_indexer_class(index_type)
    kwargs["index_type"] = index_type
    indexer = cls(**kwargs)
    if not indexer.name:
        indexer.name = index_type
    return indexer
