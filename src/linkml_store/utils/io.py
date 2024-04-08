from pathlib import Path
from typing import Iterable, Iterator, Optional, TextIO, Union

from linkml_runtime import SchemaView

from linkml_store.api.collection import OBJECT


def export_objects(
    objects: Iterable[OBJECT],
    location: Union[Path, str, TextIO],
    output_type: Optional[str],
    target_class: Optional[str] = None,
    schema_view: Optional[SchemaView] = None,
    **kwargs,
):
    """
    Export objects to a file or stream

    :param objects: objects to export
    :param location: location to export to
    :param kwargs:
    :return:
    """
    raise NotImplementedError


def import_objects_iter(
    location: Union[Path, str, TextIO], schema_view: Optional[SchemaView] = None, **kwargs
) -> Iterator[OBJECT]:
    """
    Import objects from a file or stream

    :param location:
    :param kwargs:
    :return:
    """
    raise NotImplementedError
