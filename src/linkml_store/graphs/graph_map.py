from abc import ABC
from typing import Optional

from pydantic import BaseModel

DEFAULT_IDENTIFIER_ATTRIBUTE = "id"
DEFAULT_CATEGORY_LABELS_ATTRIBUTE = "category"
DEFAULT_SUBJECT_ATTRIBUTE = "subject"
DEFAULT_PREDICATE_ATTRIBUTE = "predicate"
DEFAULT_OBJECT_ATTRIBUTE = "object"


class GraphProjection(BaseModel, ABC):
    identifier_attribute: str = DEFAULT_IDENTIFIER_ATTRIBUTE


class NodeProjection(GraphProjection):
    category_labels_attribute: Optional[str] = DEFAULT_CATEGORY_LABELS_ATTRIBUTE


class EdgeProjection(GraphProjection):
    subject_attribute: str = DEFAULT_SUBJECT_ATTRIBUTE
    predicate_attribute: str = DEFAULT_PREDICATE_ATTRIBUTE
    object_attribute: str = DEFAULT_OBJECT_ATTRIBUTE
