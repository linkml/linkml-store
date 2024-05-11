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

    :param name: the name of the indexer
    :return: the indexer class
    """
    if name not in INDEXER_CLASSES:
        raise ValueError(f"Unknown indexer class: {name}")
    return INDEXER_CLASSES[name]


def get_indexer(name: str, **kwargs) -> Indexer:
    """
    Get an indexer by name.

    :param name: the name of the indexer
    :param kwargs: additional arguments to pass to the indexer
    :return: the indexer
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    cls = get_indexer_class(name)
    kwargs["name"] = name
    indexer = cls(**kwargs)
    return indexer
