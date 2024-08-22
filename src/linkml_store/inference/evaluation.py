import logging
from collections.abc import Callable
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from linkml_store.inference import InferenceEngine
from linkml_store.utils.object_utils import select_nested

logger = logging.getLogger(__name__)


def score_match(target: Optional[Any], candidate: Optional[Any], match_function: Optional[Callable] = None) -> float:
    """
    Compute a score for a match between two objects

    >>> score_match("a", "a")
    1.0
    >>> score_match("a", "b")
    0.0
    >>> score_match("abcd", "abcde")
    0.0
    >>> score_match("a", None)
    0.0
    >>> score_match(None, "a")
    0.0
    >>> score_match(None, None)
    1.0
    >>> score_match(["a", "b"], ["a", "b"])
    1.0
    >>> score_match(["a", "b"], ["b", "a"])
    1.0
    >>> round(score_match(["a"], ["b", "a"]), 2)
    0.67
    >>> score_match({"a": 1}, {"a": 1})
    1.0
    >>> score_match({"a": 1}, {"a": 2})
    0.0
    >>> score_match({"a": 1, "b": None}, {"a": 1})
    1.0
    >>> score_match([{"a": 1, "b": 2}, {"a": 3, "b": 4}], [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    1.0
    >>> score_match([{"a": 1, "b": 4}, {"a": 3, "b": 2}], [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    0.5
    >>> def char_match(x, y):
    ...    return len(set(x).intersection(set(y))) / len(set(x).union(set(y)))
    >>> score_match("abcd", "abc", char_match)
    0.75
    >>> score_match(["abcd", "efgh"], ["ac", "gh"], char_match)
    0.5


    :param target:
    :param candidate:
    :param match_function: defaults to struct
    :return:
    """
    if target == candidate:
        return 1.0
    if target is None or candidate is None:
        return 0.0
    if isinstance(target, (set, list)) and isinstance(candidate, (set, list)):
        # create an all by all matrix using numpy
        # for each pair of elements, compute the score
        # return the average score
        score_matrix = np.array([[score_match(t, c, match_function) for c in candidate] for t in target])
        best_matches0 = np.max(score_matrix, axis=0)
        best_matches1 = np.max(score_matrix, axis=1)
        return (np.sum(best_matches0) + np.sum(best_matches1)) / (len(target) + len(candidate))
    if isinstance(target, dict) and isinstance(candidate, dict):
        keys = set(target.keys()).union(candidate.keys())
        scores = [score_match(target.get(k), candidate.get(k), match_function) for k in keys]
        return np.mean(scores)
    if match_function:
        return match_function(target, candidate)
    return 0.0


class Outcome(BaseModel):
    true_positive_count: float
    total_count: int

    @property
    def accuracy(self) -> float:
        return self.true_positive_count / self.total_count


def evaluate_predictor(
    predictor: InferenceEngine,
    target_attributes: List[str],
    feature_attributes: Optional[List[str]] = None,
    test_data: pd.DataFrame = None,
    evaluation_count: Optional[int] = 10,
    match_function: Optional[Callable] = None,
) -> Outcome:
    """
    Evaluate a predictor by comparing its predictions to the expected values in the testing data.

    :param predictor:
    :param target_attributes:
    :param feature_attributes:
    :param evaluation_count: max iterations
    :param match_function: function to use for matching
    :return:
    """
    n = 0
    tp = 0
    if test_data is None:
        test_data = predictor.testing_data.as_dataframe()
    for row in test_data.to_dict(orient="records"):
        expected_obj = select_nested(row, target_attributes)
        if feature_attributes:
            test_obj = {k: v for k, v in row.items() if k not in target_attributes}
        else:
            test_obj = row
        result = predictor.derive(test_obj)
        tp += score_match(result.predicted_object, expected_obj, match_function)
        logger.info(f"TP={tp} MF={match_function} Predicted: {result.predicted_object} Expected: {expected_obj}")
        n += 1
        if evaluation_count is not None and n >= evaluation_count:
            break
    return Outcome(true_positive_count=tp, total_count=n)


def score_text_overlap(str1: Any, str2: Any) -> float:
    """
    Compute the overlap score between two strings.

    >>> score_text_overlap("abc", "bcde")
    0.5

    :param str1:
    :param str2:
    :return:
    """
    if str1 == str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    overlap, length = find_longest_overlap(str1, str2)
    return len(overlap) / max(len(str1), len(str2))


def find_longest_overlap(str1: str, str2: str):
    """
    Find the longest overlapping substring between two strings.

    Args:
    str1 (str): The first string
    str2 (str): The second string

    Returns:
    tuple: A tuple containing the longest overlapping substring and its length

    Examples:
    >>> find_longest_overlap("hello world", "world of programming")
    ('world', 5)
    >>> find_longest_overlap("abcdefg", "defghi")
    ('defg', 4)
    >>> find_longest_overlap("python", "java")
    ('', 0)
    >>> find_longest_overlap("", "test")
    ('', 0)
    >>> find_longest_overlap("aabbcc", "ddeeff")
    ('', 0)
    >>> find_longest_overlap("programming", "PROGRAMMING")
    ('', 0)
    """
    if not str1 or not str2:
        return "", 0

    # Create a table to store lengths of matching substrings
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Variables to store the maximum length and ending position
    max_length = 0
    end_pos = 0

    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i

    # Extract the longest common substring
    start_pos = end_pos - max_length
    longest_substring = str1[start_pos:end_pos]

    return longest_substring, max_length
