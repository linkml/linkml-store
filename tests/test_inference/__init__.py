import logging
from typing import List, Optional, Union

import pandas as pd

from linkml_store.inference import InferenceEngine
from linkml_store.inference.evaluation import Outcome, evaluate_predictor, score_match
from linkml_store.utils.object_utils import select_nested

logger = logging.getLogger(__name__)


def check_accuracy(
    ie: InferenceEngine,
    target_class: Union[str, List[str]],
    threshold: Optional[float] = None,
    features: List[str] = None,
    test_data: pd.DataFrame = None,
) -> float:
    n = 0
    tp = 0
    if test_data is None:
        test_data = ie.testing_data.as_dataframe()
    for test_row in test_data.to_dict(orient="records")[0:10]:
        if isinstance(target_class, list):
            expected = select_nested(test_row, target_class)
            test_row = select_nested(test_row, features)
            # test_row = {k: v for k, v in test_row.items() if k not in target_class}
        else:
            expected = test_row.pop(target_class)
        prediction = ie.derive(test_row)
        if prediction is None:
            continue
        if isinstance(target_class, list):
            print(f"PREDICTED={prediction.predicted_object}")
            print(f"EXPECTED={expected}")
            tp += score_match(prediction.predicted_object, expected)
        else:
            if prediction.predicted_object[target_class] == expected:
                tp += 1
        n += 1
    accuracy = tp / n
    logger.info(f"Accuracy: {accuracy} ({tp}/{n}) compared to {threshold}")
    if threshold is not None:
        if accuracy < threshold:
            print(f"Accuracy: {accuracy} ({tp}/{n}) is below threshold {threshold}")
        assert accuracy >= threshold, f"Accuracy {tp}/{n} is too low"
    return accuracy


def check_accuracy2(
    ie: InferenceEngine,
    target_classes: List[str],
    threshold: Optional[float] = None,
    features: List[str] = None,
    test_data: pd.DataFrame = None,
    **kwargs,
) -> Outcome:
    if test_data is None:
        test_data = ie.testing_data.as_dataframe()
    outcome = evaluate_predictor(ie, target_classes, features, test_data, **kwargs)
    accuracy = outcome.accuracy
    if threshold is not None:
        if accuracy < threshold:
            print(f"Accuracy: {accuracy} ({outcome} is below threshold {threshold}")
        assert accuracy >= threshold, f"Accuracy {outcome} is too low"
    return outcome
