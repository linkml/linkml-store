import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from linkml_store.utils.vector_utils import mmr_diversified_search, pairwise_cosine_similarity

INDEX_ITEM = np.ndarray

logger = logging.getLogger(__name__)


class TemplateSyntaxEnum(str, Enum):
    """
    Template syntax types.
    """

    jinja2 = "jinja2"
    fstring = "fstring"


class Indexer(BaseModel):
    """
    An indexer operates on a collection in order to search for objects.

    You should use a subcllass of this; this can be looked up dynqamically:

    >>> from linkml_store.index import get_indexer
    >>> indexer = get_indexer("simple")

    You can customize how objects are indexed by passing in a text template.
    For example, if your collection has objects with "name" and "profession" attributes,
    you can index them as "{name} {profession}".

    >>> indexer = get_indexer("simple", text_template="{name} :: {profession}")

    By default, python fstrings are assumed.

    We can test this works using the :ref:`object_to_text` method (normally
    you would never need to call this directly, but it's useful for testing):

    >>> obj = {"name": "John", "profession": "doctor"}
    >>> indexer.object_to_text(obj)
    'John :: doctor'

    You can also use Jinja2 templates; this gives more flexibility and logic,
    e.g. conditional formatting:

    >>> tmpl = "{{name}}{% if profession %} :: {{profession}}{% endif %}"
    >>> indexer = get_indexer("simple", text_template=tmpl, text_template_syntax=TemplateSyntaxEnum.jinja2)
    >>> indexer.object_to_text(obj)
    'John :: doctor'
    >>> indexer.object_to_text({"name": "John"})
    'John'

    You can also specify which attributes to index:

    >>> indexer = get_indexer("simple", index_attributes=["name"])
    >>> indexer.object_to_text(obj)
    'John'

    The purpose of an indexer is to translate a collection of objects into a collection of objects
    such as vectors for purposes such as search. Unless you are implementing your own indexer, you
    generally don't need to use the methods that return vectors, but we can examine their behavior
    to get a sense of how they work.

    >>> vectors = indexer.objects_to_vectors([{"name": "Aardvark"}, {"name": "Aardwolf"}, {"name": "Zesty"}])
    >>> assert pairwise_cosine_similarity(vectors[0], vectors[1]) > pairwise_cosine_similarity(vectors[0], vectors[2])

    Note you should consult the documentation for the specific indexer you are using for more details on
    how text is converted to vectors.

    """

    name: Optional[str] = None
    index_type: Optional[str] = None
    index_function: Optional[Callable] = None
    distance_function: Optional[Callable] = None
    index_attributes: Optional[List[str]] = None
    text_template: Optional[str] = None
    text_template_syntax: Optional[TemplateSyntaxEnum] = None
    filter_nulls: Optional[bool] = True
    vector_default_length: Optional[int] = 1000
    index_field: Optional[str] = "__index__"

    def object_to_vector(self, obj: Dict[str, Any]) -> INDEX_ITEM:
        """
        Convert an object to an indexable object

        :param obj:
        :return:
        """
        return self.text_to_vector(self.object_to_text(obj))

    def objects_to_vectors(self, objs: List[Dict[str, Any]]) -> List[INDEX_ITEM]:
        """
        Convert a list of objects to indexable objects

        :param objs:
        :return: list of vectors
        """
        return self.texts_to_vectors([self.object_to_text(obj) for obj in objs])

    def texts_to_vectors(self, texts: List[str], cache: bool = None, **kwargs) -> List[INDEX_ITEM]:
        """
        Convert a list of texts to indexable objects

        :param texts:
        :return:
        """
        return [self.text_to_vector(text, cache=cache, **kwargs) for text in texts]

    def text_to_vector(self, text: str, cache: bool = None, **kwargs) -> INDEX_ITEM:
        """
        Convert a text to an indexable object

        :param text:
        :param cache:
        :return:
        """
        raise NotImplementedError

    def object_to_text(self, obj: Dict[str, Any]) -> str:
        """
        Convert an object to a text representation

        :param obj:
        :return:
        """
        if self.index_attributes:
            if len(self.index_attributes) == 1 and not self.text_template:
                return str(obj[self.index_attributes[0]])
            obj = {k: v for k, v in obj.items() if k in self.index_attributes}
        if self.filter_nulls:
            obj = {k: v for k, v in obj.items() if v is not None}
        if self.text_template:
            syntax = self.text_template_syntax
            if not syntax:
                if "{%" in self.text_template or "{{" in self.text_template:
                    logger.info("Detected Jinja2 syntax in text template")
                    syntax = TemplateSyntaxEnum.jinja2
            if not syntax:
                syntax = TemplateSyntaxEnum.fstring
            if syntax == TemplateSyntaxEnum.jinja2:
                from jinja2 import Template

                template = Template(self.text_template)
                return template.render(**obj)
            elif syntax == TemplateSyntaxEnum.fstring:
                return self.text_template.format(**obj)
            else:
                raise NotImplementedError(f"Cannot handle template syntax: {syntax}")
        return str(obj)

    def search(
        self,
        query: str,
        vectors: List[Tuple[str, INDEX_ITEM]],
        limit: Optional[int] = None,
        mmr_relevance_factor: Optional[float] = None,
    ) -> List[Tuple[float, Any]]:
        """
        Use the indexer to search against a database of vectors.

        Note: this is a low-level method, typically you would use the :ref:`search` method on a :ref:`Collection`.

        :param query: The query string to search for
        :param vectors: A list of indexed items, where each item is a tuple of (id, vector)
        :param limit: The maximum number of results to return (optional)
        :return: A list of item IDs or objects that match the query
        """

        # Convert the query string to a vector
        query_vector = self.text_to_vector(query, cache=False)

        if mmr_relevance_factor is not None:
            vlist = [v for _, v in vectors]
            idlist = [id for id, _ in vectors]
            sorted_indices = mmr_diversified_search(
                query_vector, vlist, relevance_factor=mmr_relevance_factor, top_n=limit
            )
            results = []
            # TODO: this is inefficient when limit is high
            for i in range(limit):
                if i >= len(sorted_indices):
                    break
                pos = sorted_indices[i]
                score = pairwise_cosine_similarity(query_vector, vlist[pos])
                results.append((score, idlist[pos]))
            return results

        distances = []

        # Iterate over each indexed item
        for item_id, item_vector in vectors:
            # Calculate the Euclidean distance between the query vector and the item vector
            # distance = 1-np.linalg.norm(query_vector - item_vector)
            distance = pairwise_cosine_similarity(query_vector, item_vector)
            distances.append((distance, item_id))

        # Sort the distances in ascending order
        distances.sort(key=lambda x: -x[0])

        # Limit the number of results if specified
        if limit is not None:
            distances = distances[:limit]

        return distances
