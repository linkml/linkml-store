import json
from copy import deepcopy
from typing import Any, Dict, List, Union

from pydantic import BaseModel


def object_path_update(
    obj: Union[BaseModel, Dict[str, Any]], path: str, value: Any
) -> Union[BaseModel, Dict[str, Any]]:
    """
    Updates a nested object based on a path description and a value. The path to the
    desired field is given in dot and bracket notation (e.g., 'a[0].b.c[1]').

    :param obj: The dictionary object to be updated.
    :type obj: Dict[str, Any]
    :param path: The path string indicating where to place the value within the object.
    :type path: str
    :param value: The value to be set at the specified path.
    :type value: Any
    :return: None. This function modifies the object in-place.
    :rtype: None

    **Example**::

    >>> data = {}
    >>> object_path_update(data, 'persons[0].foo.bar', 1)
    {'persons': [{'foo': {'bar': 1}}]}
    """
    if isinstance(obj, BaseModel):
        typ = type(obj)
        obj = obj.dict()
        obj = object_path_update(obj, path, value)
        return typ(**obj)
    obj = deepcopy(obj)
    ret_obj = obj
    parts = path.split(".")
    for part in parts[:-1]:
        if "[" in part:
            key, index = part[:-1].split("[")
            index = int(index)
            # obj = obj.setdefault(key, [{} for _ in range(index+1)])
            obj = obj.setdefault(key, [])
            while len(obj) <= index:
                obj.append({})
            obj = obj[index]
        else:
            obj = obj.setdefault(part, {})
    last_part = parts[-1]
    if "[" in last_part:
        key, index = last_part[:-1].split("[")
        index = int(index)
        if key not in obj or not isinstance(obj[key], list):
            obj[key] = [{} for _ in range(index + 1)]
        obj[key][index] = value
    else:
        obj[last_part] = value
    return ret_obj


def parse_update_expression(expr: str) -> Union[tuple[str, Any], None]:
    """
    Parse a string expression of the form 'path.to.field=value' into a path and a value.

    :param expr:
    :return:
    """
    try:
        path, val = expr.split("=", 1)
        val = json.loads(val)
    except ValueError:
        return None
    return path, val


def clean_empties(value: Union[Dict, List]) -> Any:
    if isinstance(value, dict):
        value = {k: v for k, v in ((k, clean_empties(v)) for k, v in value.items()) if v is not None}
    elif isinstance(value, list):
        value = [v for v in (clean_empties(v) for v in value) if v is not None]
    return value
