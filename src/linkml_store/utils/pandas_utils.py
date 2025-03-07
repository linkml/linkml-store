import logging
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Recursively flatten a nested dictionary.

    Args:
        d (Dict[str, Any]): The dictionary to flatten.
        parent_key (str): The parent key for nested dictionaries.
        sep (str): The separator to use between keys.

    Returns:
        Dict[str, Any]: A flattened dictionary.

    >>> flatten_dict({'a': 1, 'b': {'c': 2, 'd': {'e': 3}}})
    {'a': 1, 'b.c': 2, 'b.d.e': 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def nested_objects_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of nested objects to a flattened pandas DataFrame.

    Args:
        data (List[Dict[str, Any]]): A list of nested dictionaries.

    Returns:
        pd.DataFrame: A flattened DataFrame.

    >>> data = [
    ...     {"person": {"name": "Alice", "age": 30}, "job": {"title": "Engineer", "salary": 75000}},
    ...     {"person": {"name": "Bob", "age": 35}, "job": {"title": "Manager", "salary": 85000}}
    ... ]
    >>> df = nested_objects_to_dataframe(data)
    >>> df.columns.tolist()
    ['person.name', 'person.age', 'job.title', 'job.salary']
    >>> df['person.name'].tolist()
    ['Alice', 'Bob']
    """
    flattened_data = [flatten_dict(item) for item in data]
    return pd.DataFrame(flattened_data)


def facet_summary_to_dataframe_unmelted(
    facet_summary: Dict[Union[str, Tuple[str, ...]], List[Tuple[Union[str, Tuple[str, ...]], int]]],
) -> pd.DataFrame:
    rows = []

    for facet_type, facet_data in facet_summary.items():
        if isinstance(facet_type, str):
            # Single facet type
            for category, value in facet_data:
                rows.append({facet_type: category, "Value": value})
        else:
            # Multiple facet types
            for cat_val_tuple in facet_data:
                if len(cat_val_tuple) == 2:
                    categories, value = cat_val_tuple
                else:
                    categories, value = cat_val_tuple[:-1], cat_val_tuple[-1]
                row = {"Value": value}
                for i, facet in enumerate(facet_type):
                    logger.debug(f"FT={facet_type} i={i} Facet: {facet}, categories: {categories}")
                    row[facet] = categories[i] if len(categories) > i else None
                rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure all columns are present, fill with None if missing
    all_columns = set(col for facet in facet_summary.keys() for col in (facet if isinstance(facet, tuple) else [facet]))
    for col in all_columns:
        if col not in df.columns:
            df[col] = None

    # Move 'Value' to the end
    cols = [col for col in df.columns if col != "Value"] + ["Value"]
    df = df[cols]

    return df
