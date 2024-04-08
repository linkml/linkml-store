from typing import Callable, Optional

from pydantic import BaseModel


class Index(BaseModel):
    """
    An index operates on a collection in order to search for objects.
    """

    name: str
    index_function: Optional[Callable] = None
    distance_function: Optional[Callable] = None
