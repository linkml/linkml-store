import operator
from typing import Any, Callable, Dict

MONGO_OPERATORS = {
    "$eq": operator.eq,
    "$ne": operator.ne,
    "$gt": operator.gt,
    "$gte": operator.ge,
    "$lt": operator.lt,
    "$lte": operator.le,
    "$in": lambda a, b: any(x in b for x in (a if isinstance(a, list) else [a])),
    "$nin": lambda a, b: all(x not in b for x in (a if isinstance(a, list) else [a])),
}


def mongo_query_to_match_function(where: Dict[str, Any]) -> Callable[[Dict[str, Any]], bool]:
    """
    Convert a MongoDB-style query to a matching function.

    >>> query = {"name": "foo", "age": {"$gt": 25}}
    >>> matcher = mongo_query_to_match_function(query)
    >>> matcher({"name": "foo", "age": 30})
    True
    >>> matcher({"name": "foo", "age": 20})
    False
    >>> matcher({"name": "bar", "age": 30})
    False

    >>> nested_query = {"nested.job": "engineer", "skills": {"$in": ["python", "mongodb"]}}
    >>> nested_matcher = mongo_query_to_match_function(nested_query)
    >>> nested_matcher({"nested": {"job": "engineer"}, "skills": ["python", "javascript"]})
    True
    >>> nested_matcher({"nested": {"job": "designer"}, "skills": ["python", "mongodb"]})
    False
    >>> nested_matcher({"nested": {"job": "engineer"}, "skills": ["java", "c++"]})
    False

    >>> complex_query = {"name": "foo", "age": {"$gte": 25, "$lt": 40}, "nested.salary": {"$gt": 50000}}
    >>> complex_matcher = mongo_query_to_match_function(complex_query)
    >>> complex_matcher({"name": "foo", "age": 30, "nested": {"salary": 60000}})
    True
    >>> complex_matcher({"name": "foo", "age": 45, "nested": {"salary": 70000}})
    False
    >>> complex_matcher({"name": "foo", "age": 35, "nested": {"salary": 40000}})
    False

    A nested dict of conditions applies to the nested object:

    >>> dict_matcher = mongo_query_to_match_function({"nested": {"job": "engineer", "salary": {"$gt": 50000}}})
    >>> dict_matcher({"nested": {"job": "engineer", "salary": 60000}})
    True
    >>> dict_matcher({"job": "engineer", "salary": 60000, "nested": {"job": "designer", "salary": 1}})
    False

    >>> invalid_query = {"age": {"$invalid": 25}}
    >>> invalid_matcher = mongo_query_to_match_function(invalid_query)
    >>> invalid_matcher({"age": 30})
    Traceback (most recent call last):
    ...
    ValueError: Unsupported operator: $invalid
    """
    if where is None:
        where = {}

    def matches(obj: Dict[str, Any]) -> bool:
        def check_condition(key: str, condition: Any) -> bool:
            if isinstance(condition, dict) and any(k.startswith("$") for k in condition.keys()):
                for op, value in condition.items():
                    if op in MONGO_OPERATORS:
                        if not MONGO_OPERATORS[op](get_nested_value(obj, key), value):
                            return False
                    else:
                        raise ValueError(f"Unsupported operator: {op}")
            elif isinstance(condition, dict):
                # conditions on the attributes of a nested object
                return all(check_condition(f"{key}.{k}", v) for k, v in condition.items())
            else:
                return get_nested_value(obj, key) == condition
            return True

        def get_nested_value(obj: Dict[str, Any], key: str) -> Any:
            parts = key.split(".")
            for part in parts:
                if isinstance(obj, dict):
                    obj = obj.get(part)
                else:
                    return None
            return obj

        return all(check_condition(k, v) for k, v in where.items())

    return matches
