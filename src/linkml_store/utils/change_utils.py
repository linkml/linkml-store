from typing import List

from linkml_store.api.collection import OBJECT


def insert_operation_to_patches(objs: List[OBJECT], **kwargs):
    """
    Translate a list of objects to a list of patches for insertion.

    Note: inserts are always treated as being at the start of a list

    :param objs: objects to insert
    :param kwargs: additional arguments
    """
    patches = []
    for obj in objs:
        patches.append({"op": "add", "path": "/0", "value": obj})
