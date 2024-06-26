from typing import Any, Dict, List, Optional, TypedDict

import jsonpatch


class PatchDict(TypedDict):
    op: str
    path: str
    value: Optional[Any]
    _from: Optional[str]


def apply_patches(obj: Any, patches: List[PatchDict], primary_key: Optional[str] = None, in_place=False) -> Any:
    """
    Apply a set of patches to an object.

    If the object is a list, the primary key must be specified.

    >>> objs = [{'id': 'F1', 'name': 'Cheese'}, {'id': 'F2', 'name': 'Bread'}]
    >>> patches = [{'op': 'replace', 'path': '/F1/name', 'value': 'Toast'}]
    >>> apply_patches(objs, patches, primary_key='id')
    [{'id': 'F1', 'name': 'Toast'}, {'id': 'F2', 'name': 'Bread'}]

    :param obj: object to patch
    :param patches: list of patches, conforming to the JSON Patch format
    :param primary_key: key to use as the primary key for the objects (if obj is a list)
    :param in_place: whether to apply the patches in place
    :return:
    """
    if isinstance(obj, dict):
        patch_obj = jsonpatch.JsonPatch(patches)
        return patch_obj.apply(obj, in_place=in_place)
    elif isinstance(obj, list):
        if not primary_key:
            raise ValueError("Primary key must be specified for list objects")
        return apply_patches_to_list(obj, patches, primary_key, in_place=in_place)
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")


def apply_patches_to_list(
    objects: List[Dict[str, Any]], patches: List[PatchDict], primary_key: str, in_place=False
) -> List[Dict[str, Any]]:
    """
    Apply a set of patches to a list of objects.



    :param objects: list of objects
    :param patches: list of patches, conforming to the JSON Patch format
    :param primary_key: key to use as the primary key for the objects
    :param in_place: whether to apply the patches in place
    :return:
    """
    objs_as_dict = {obj[primary_key]: obj for obj in objects}
    result = apply_patches_to_keyed_list(objs_as_dict, patches, in_place=in_place)
    return list(result.values())


def apply_patches_to_keyed_list(
    objs_as_dict: Dict[str, Dict[str, Any]], patches: List[PatchDict], in_place=False
) -> Dict[str, Dict[str, Any]]:
    """
    Apply a set of patches to a list of objects, where the objects are keyed by a primary key

    :param objs_as_dict:
    :param patches:
    :param in_place:
    :return:
    """
    patch_obj = jsonpatch.JsonPatch(patches)
    result = patch_obj.apply(objs_as_dict, in_place=in_place)
    return result


def patches_from_objects_lists(
    src_objs: List[Dict[str, Any]], dst_objs: List[Dict[str, Any]], primary_key: str, exclude_none=True
) -> List[PatchDict]:
    """
    Generate a set of patches to transform src_objs into tgt_objs.

    >>> src_objs = [{'id': 'F1', 'name': 'Cheese'}, {'id': 'F2', 'name': 'Bread'}]
    >>> tgt_objs = [{'id': 'F1', 'name': 'Toast'}, {'id': 'F2', 'name': 'Bread'}]
    >>> patches_from_objects_lists(src_objs, tgt_objs, primary_key='id')
    [{'op': 'replace', 'path': '/F1/name', 'value': 'Toast'}]

    by default exclude_none is True, so None values are excluded from the patch

    >>> tgt_objs = [{'id': 'F1', 'name': 'Toast'}, {'id': 'F2', 'name': None}]
    >>> patches_from_objects_lists(src_objs, tgt_objs, primary_key='id')
    [{'op': 'replace', 'path': '/F1/name', 'value': 'Toast'}, {'op': 'remove', 'path': '/F2/name'}]

    if exclude_none is False, None values are treated as being set to None

    >>> patches_from_objects_lists(src_objs, tgt_objs, primary_key='id', exclude_none=False)
    [{'op': 'replace', 'path': '/F1/name', 'value': 'Toast'}, {'op': 'replace', 'path': '/F2/name', 'value': None}]

    See also: `<https://github.com/orgs/linkml/discussions/1975>`_

    Note the patches are sorted deterministically, first by path, then by operation.
    This helps ensure operations on the same object are grouped together

    :param src_objs: source objects
    :param dst_objs: target objects
    :param primary_key: key to use as the primary key for the objects
    :param exclude_none: whether to exclude None values from the patch
    :return:
    """
    src_objs_as_dict = {obj[primary_key]: obj for obj in src_objs}
    dst_objs_as_dict = {obj[primary_key]: obj for obj in dst_objs}
    if exclude_none:
        src_objs_as_dict = {k: remove_nones(v) for k, v in src_objs_as_dict.items()}
        dst_objs_as_dict = {k: remove_nones(v) for k, v in dst_objs_as_dict.items()}
    patch_obj = jsonpatch.JsonPatch.from_diff(src_objs_as_dict, dst_objs_as_dict)
    pl = patch_obj.patch
    return sorted(pl, key=lambda x: (x["path"], x["op"]))


def remove_nones(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove None values from a dictionary.

    :param obj:
    :return:
    """
    return {k: v for k, v in obj.items() if v is not None}
