import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

INDEX_ITEM = np.ndarray

logger = logging.getLogger(__name__)


class TemplateSyntaxEnum(str, Enum):
    """
    Template syntax types.
    """

    jinja2 = "jinja2"
    fstring = "fstring"


def cosine_similarity(vector1, vector2) -> float:
    """
    Calculate the cosine similarity between two vectors

    :param vector1:
    :param vector2:
    :return:
    """
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2)


class Indexer(BaseModel):
    """
    An indexer operates on a collection in order to search for objects.
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
        self, query: str, vectors: List[Tuple[str, INDEX_ITEM]], limit: Optional[int] = None
    ) -> List[Tuple[float, Any]]:
        """
        Search the index for a query string

        :param query: The query string to search for
        :param vectors: A list of indexed items, where each item is a tuple of (id, vector)
        :param limit: The maximum number of results to return (optional)
        :return: A list of item IDs or objects that match the query
        """

        # Convert the query string to a vector
        query_vector = self.text_to_vector(query, cache=False)

        distances = []

        # Iterate over each indexed item
        for item_id, item_vector in vectors:
            # Calculate the Euclidean distance between the query vector and the item vector
            # distance = 1-np.linalg.norm(query_vector - item_vector)
            distance = cosine_similarity(query_vector, item_vector)
            distances.append((distance, item_id))

        # Sort the distances in ascending order
        distances.sort(key=lambda x: -x[0])

        # Limit the number of results if specified
        if limit is not None:
            distances = distances[:limit]

        return distances
