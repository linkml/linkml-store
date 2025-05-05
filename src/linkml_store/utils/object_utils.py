import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

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
        obj = obj.model_dump(exclude_none=True)
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
            if part in obj and obj[part] is None:
                del obj[part]
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


def object_path_get(obj: Union[BaseModel, Dict[str, Any]], path: str, default_value=None) -> Any:
    """
    Retrieves a value from a nested object based on a path description. The path to the
    desired field is given in dot and bracket notation (e.g., 'a[0].b.c[1]').

    :param obj: The dictionary object to be updated.
    :type obj: Dict[str, Any]
    :param path: The path string indicating where to place the value within the object.
    :type path: str
    :return: The value at the specified path.
    :rtype: Any

    **Example**::

    >>> data = {'persons': [{'foo': {'bar': 1}}]}
    >>> object_path_get(data, 'persons[0].foo.bar')
    1
    >>> object_path_get(data, 'persons[0].foo')
    {'bar': 1}
    >>> object_path_get({}, 'not there', "NA")
    'NA'
    """
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    parts = path.split(".")
    for part in parts:
        if "[" in part:
            key, index = part[:-1].split("[")
            index = int(index)
            if key in obj and obj[key] is not None:
                obj = obj[key][index]
            else:
                return default_value
        else:
            if isinstance(obj, list):
                obj = [v1.get(part, default_value) for v1 in obj]
            else:
                obj = obj.get(part, default_value)
    return obj


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


def select_nested(data: dict, paths: List[Union[str, List[str]]], current_path=None) -> Optional[dict]:
    """
    Select nested attributes from a complex dictionary based on selector strings.

    Args:
    data (dict): The input nested dictionary.
    paths (list): A list of selector strings.

    Returns:
    dict: A new dictionary with the same structure, but only the selected attributes.

    Example:
    >>> data = {
    ...     "person": {
    ...         "name": "John Doe",
    ...         "age": 30,
    ...         "address": {
    ...             "street": "123 Main St",
    ...             "city": "Anytown",
    ...             "country": "USA"
    ...         },
    ...         "phones": [
    ...             {"type": "home", "number": "555-1234"},
    ...             {"type": "work", "number": "555-5678"}
    ...         ]
    ...     },
    ...     "company": {
    ...         "name": "Acme Inc",
    ...         "location": "New York"
    ...     }
    ... }
    >>> select_nested(data, ["person.address.street", "person.address.city"])
    {'person': {'address': {'street': '123 Main St', 'city': 'Anytown'}}}
    >>> select_nested(data, ["person.phones.number", "person.phones.type"])
    {'person': {'phones': [{'type': 'home', 'number': '555-1234'}, {'type': 'work', 'number': '555-5678'}]}}
    >>> select_nested(data, ["person"])
    {'person': {'name': 'John Doe', 'age': 30, 'address': {'street': '123 Main St', 'city': 'Anytown',
     'country': 'USA'}, 'phones': [{'type': 'home', 'number': '555-1234'}, {'type': 'work', 'number': '555-5678'}]}}
    >>> select_nested(data, ["person.phones.type"])
    {'person': {'phones': [{'type': 'home'}, {'type': 'work'}]}}
    """
    if current_path is None:
        current_path = []
    matching_paths = []
    if not paths:
        raise ValueError("No paths provided")
    for path in paths:
        if isinstance(path, str):
            path = path.split(".")
        if path == current_path:
            return data
        if path[: len(current_path)] == current_path:
            matching_paths.append(path)
    if not matching_paths:
        return None
    if isinstance(data, dict):
        new_obj = {k: select_nested(v, matching_paths, current_path + [k]) for k, v in data.items()}
        new_obj = {k: v for k, v in new_obj.items() if v is not None}
        return new_obj
    if isinstance(data, list):
        new_obj = [select_nested(v, matching_paths, current_path + []) for i, v in enumerate(data)]
        new_obj = [v for v in new_obj if v is not None]
        return new_obj
    return data
