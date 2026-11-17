import operator
from typing import Any, Callable, Dict, List


def _compare(op: Callable[[Any, Any], bool]) -> Callable[[Any, Any], bool]:
    """Wrap a range comparison so that missing values and incomparable types never match."""

    def compare(a: Any, b: Any) -> bool:
        if a is None:
            return False
        try:
            return op(a, b)
        except TypeError:
            return False

    return compare


MONGO_OPERATORS = {
    "$eq": operator.eq,
    "$ne": operator.ne,
    "$gt": _compare(operator.gt),
    "$gte": _compare(operator.ge),
    "$lt": _compare(operator.lt),
    "$lte": _compare(operator.le),
    "$in": lambda a, b: any(x in b for x in (a if isinstance(a, list) else [a])),
    "$nin": lambda a, b: all(x not in b for x in (a if isinstance(a, list) else [a])),
}

# operators that match when *no* value satisfies the positive form
NEGATED_OPERATORS = {"$ne": "$eq", "$nin": "$in"}


def path_values(obj: Any, parts: List[str]) -> List[Any]:
    """
    All values reached by following a dotted path, traversing into lists as MongoDB does.

    >>> path_values({"a": [{"b": 1}, {"b": 2}, {"c": 3}]}, ["a", "b"])
    [1, 2]
    >>> path_values({"a": {"b": [1, 2]}}, ["a", "b"])
    [[1, 2]]
    >>> path_values({"a": 1}, ["a", "b"])
    []
    """
    if not parts:
        return [obj]
    if isinstance(obj, list):
        return [v for item in obj for v in path_values(item, parts)]
    if isinstance(obj, dict) and parts[0] in obj:
        return path_values(obj[parts[0]], parts[1:])
    return []


def _candidates(values: List[Any]) -> List[Any]:
    """Each value, plus the elements of any value that is a list (MongoDB array semantics)."""
    out = []
    for v in values:
        out.append(v)
        if isinstance(v, list):
            out.extend(v)
    return out


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

    Missing values never satisfy a range comparison:

    >>> matcher({"name": "foo"})
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

    As in MongoDB, a scalar condition on a list field matches any element, paths traverse lists,
    and a dictionary condition (with no operators) matches a sub-object exactly:

    >>> mongo_query_to_match_function({"skills": "python"})({"skills": ["python", "java"]})
    True
    >>> mongo_query_to_match_function({"rels.type": "friend"})({"rels": [{"type": "friend"}]})
    True
    >>> mongo_query_to_match_function({"meta": {"notes": "x"}})({"meta": {"notes": "x"}})
    True
    >>> mongo_query_to_match_function({"meta": {"notes": "x"}})({"meta": {"notes": "x", "other": 1}})
    False
    >>> mongo_query_to_match_function({"skills": {"$ne": "python"}})({"skills": ["python", "java"]})
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
        def check_operator(values: List[Any], op: str, value: Any) -> bool:
            if op not in MONGO_OPERATORS:
                raise ValueError(f"Unsupported operator: {op}")
            if op in NEGATED_OPERATORS:
                return not check_operator(values, NEGATED_OPERATORS[op], value)
            if not values:
                # a missing field behaves as null
                return MONGO_OPERATORS[op](None, value)
            return any(MONGO_OPERATORS[op](v, value) for v in _candidates(values))

        def check_condition(key: str, condition: Any) -> bool:
            values = path_values(obj, key.split("."))
            if isinstance(condition, dict) and any(k.startswith("$") for k in condition.keys()):
                return all(check_operator(values, op, value) for op, value in condition.items())
            if condition is None:
                return not values or any(v is None for v in values)
            return any(v == condition for v in _candidates(values))

        return all(check_condition(k, v) for k, v in where.items())

    return matches
