import json
import logging
import random
from io import StringIO
from random import randint
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from linkml_runtime.utils.eval_utils import eval_expr
from sklearn.preprocessing import OneHotEncoder

from linkml_store.api.client import Client
from linkml_store.inference import InferenceConfig, get_inference_engine
from linkml_store.inference.implementations.rule_based_inference_engine import RuleBasedInferenceEngine
from linkml_store.inference.implementations.sklearn_inference_engine import SklearnInferenceEngine
from linkml_store.inference.inference_engine import InferenceEngine, ModelSerialization
from linkml_store.utils.format_utils import Format
from tests import INPUT_DIR, OUTPUT_DIR
from tests.test_inference import check_accuracy, check_accuracy2

MODEL_FILE_PATH = OUTPUT_DIR / "model.joblib"
RULE_BASED_MODEL_FILE_PATH = OUTPUT_DIR / "sklean-export.rulebase.yaml"

logger = logging.getLogger(__name__)


def get_dataset(name: str, version: Optional[int] = 2) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Get a dataset from OpenML.

    :param name:
    :param version:
    :return:
    """
    from sklearn.datasets import fetch_openml

    dataset = fetch_openml(name=name, version=version, as_frame=True)
    df = pd.concat([dataset.data, dataset.target], axis=1)
    return df, list(dataset.data.columns), ["class"]


def roundtrip(ie: InferenceEngine) -> SklearnInferenceEngine:
    ie.save_model(MODEL_FILE_PATH)
    ie2 = ie.load_model(MODEL_FILE_PATH)
    assert isinstance(ie2, SklearnInferenceEngine)
    return ie2


def make_rule_based(ie: InferenceEngine) -> RuleBasedInferenceEngine:
    ie.export_model(RULE_BASED_MODEL_FILE_PATH, model_serialization=ModelSerialization.RULE_BASED)
    ie2 = RuleBasedInferenceEngine.load_model(RULE_BASED_MODEL_FILE_PATH)
    return ie2


def test_inference_basic():
    """
    Test the sklearn inference engine using iris dataset.

    Also tests derivation of rules from the classification model
    (decision tree) and creation and application of a rule-based
    inference engine.
    """
    client = Client()
    db = client.attach_database("duckdb", alias="test")
    db.import_database(INPUT_DIR / "iris.jsonl", Format.JSONL, collection_name="iris")
    assert db.list_collection_names() == ["iris"]
    collection = db.get_collection("iris")
    assert collection.find({}).num_rows == 100  # TODO
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    config = InferenceConfig(target_attributes=["species"], feature_attributes=features)
    ie = get_inference_engine("sklearn", config=config)
    assert isinstance(ie.config, InferenceConfig)
    # don't randomize split as this makes test non-deterministic
    ie.load_and_split_data(collection, randomize=False)
    ie.initialize_model()
    assert isinstance(ie, SklearnInferenceEngine)
    assert set(ie.encoders.keys()) == {"species"}
    q = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    prediction = ie.derive(q)
    assert prediction.predicted_object["species"] == "setosa"
    check_accuracy(ie, "species", threshold=0.9)
    ie2 = roundtrip(ie)
    check_accuracy(ie2, "species", threshold=0.9, test_data=ie.testing_data.as_dataframe())
    assert isinstance(ie2.config, InferenceConfig)
    io = StringIO()
    ie.export_model(io, model_serialization=ModelSerialization.LINKML_EXPRESSION)
    expr = io.getvalue()
    assert eval_expr(expr, **q) == "setosa"
    io = StringIO()
    ie.export_model(io, model_serialization=ModelSerialization.LINKML_EXPRESSION)
    logger.info(f"RULES: {io.getvalue()}")
    rule_engine = get_inference_engine("rulebased")
    rule_engine.import_model_from(ie)
    assert isinstance(rule_engine.config, InferenceConfig)
    prediction = rule_engine.derive(q)
    assert prediction.predicted_object["species"] == "setosa"
    check_accuracy(rule_engine, "species", threshold=0.9, test_data=ie.testing_data.as_dataframe())
    rbie = make_rule_based(ie)
    assert isinstance(rbie.config, InferenceConfig)
    prediction = rbie.derive(q)
    assert prediction.predicted_object["species"] == "setosa"


@pytest.mark.parametrize("seed", [42])
def test_inference_mixed(seed):
    """
    Test the sklearn inference engine using mixed categorical and numerical data.
    """
    client = Client()
    df, features, targets = get_dataset("adult", 2)
    df = df.replace({np.nan: None})
    # https://github.com/pandas-dev/pandas/issues/58230
    rows = json.loads(df.to_json(orient="records"))
    for row in rows:
        assert isinstance(row["capital-gain"], int), "expected all ints for capital-gain"
    db = client.attach_database("duckdb", alias="test")
    db.store({"data": rows})
    collection = db.get_collection("data")
    for row in collection.find({}, limit=-1).rows:
        assert isinstance(row["capital-gain"], int), "expected all ints for capital-gain after storage in db"
    config = InferenceConfig(target_attributes=targets, feature_attributes=features, random_seed=seed)
    ie = get_inference_engine("sklearn", config=config)
    assert isinstance(ie, SklearnInferenceEngine)
    ie.load_and_split_data(collection)
    ie.initialize_model()
    logger.info(f"Features after encoding: {ie.transformed_features}")
    assert {"age", "workclass_Private"}.difference(
        ie.transformed_features
    ) == set(), "expected transform of categorical"
    assert isinstance(ie.encoders["sex"], OneHotEncoder)
    assert "capital-gain" not in ie.encoders
    logger.info(f"Targets after encoding: {ie.transformed_targets}")
    assert set(ie.transformed_targets) == {"<=50K", ">50K"}, "no need for one-hot for target"
    check_accuracy(ie, "class", threshold=0.4)
    io = StringIO()
    ie.export_model(io, model_serialization=ModelSerialization.LINKML_EXPRESSION)
    # print(f"RULES: {io.getvalue()}")
    rule_engine = get_inference_engine("rulebased")
    rule_engine.import_model_from(ie)
    check_accuracy(rule_engine, "class", threshold=0.1, test_data=ie.testing_data.as_dataframe())


@pytest.mark.parametrize("seed", [42, 101])
def test_nested_data(seed):
    """
    Test the sklearn inference engine with nested data.

    This uses a trivial dataset where the target is the age of a twin,
    and the features are the age and name of a person.

    E.g.

    {"person": {"name": "Person 1", "age": 21}, "twin": {"name": "Twin 1", "age": 21}}
    """
    # local_random = random.Random(seed)
    client = Client()
    tgt = "twin.age"
    n = 100

    def _age(i: int) -> int:
        return i % 5 + 20

    objects = [
        {"person": {"name": f"Person {i}", "age": _age(i)}, "twin": {"name": f"Twin {i}", "age": _age(i)}}
        for i in range(n)
    ]
    db = client.attach_database("duckdb", alias="test")
    db.store({"data": objects})
    collection = db.get_collection("data")
    config = InferenceConfig(target_attributes=[tgt], feature_attributes=["person.name", "person.age"])
    config.random_seed = seed
    # TODO: infer feature cols for nested
    # config = InferenceConfig(target_attributes=[tgt])
    ie = get_inference_engine("sklearn", config=config)
    assert isinstance(ie, SklearnInferenceEngine)
    ie.load_and_split_data(collection)
    ie.initialize_model()
    assert ie.transformed_features == ["person.age"], "expected filtering of non-informative attributes"
    assert not ie.encoders, "expected no encoders as filtered feature and target are ints"
    check_accuracy(ie, tgt, threshold=0.99, test_data=ie.testing_data.as_dataframe(flattened=True))
    print(f"Features after encoding: {ie.transformed_features}")
    print(f"Targets after encoding: {ie.transformed_targets}")
    print(f"Encoders: {ie.encoders}")
    io = StringIO()
    ie.export_model(io, model_serialization=ModelSerialization.LINKML_EXPRESSION)
    print(f"RULES: {io.getvalue()}")
    rule_engine = get_inference_engine("rulebased")
    rule_engine.import_model_from(ie)
    # TODO: test the rule engine once eval_utils supports nested objects


def test_unseen_categories():
    """
    Test the sklearn inference engine when test data has unseen categories.

    We expect graceful degradation - inability to predict the unseen category

    :return:
    """
    client = Client()
    tgt = "class"
    n = 100

    def _category(i: int) -> str:
        return "x" + str(i % 5)

    objects = [{"feature": _category(i), tgt: _category(i)} for i in range(n)]
    test_data = [
        {"feature": "x99", tgt: "x99"},
        {"feature": "x1", tgt: "x99"},
        {"feature": "x99", tgt: "x1"},
        {"feature": None, tgt: "x1"},
        {tgt: "x1"},
        {},
        {"feature": None, tgt: None},
    ]

    db = client.attach_database("duckdb", alias="test")
    db.store({"data": objects})
    collection = db.get_collection("data")
    config = InferenceConfig(target_attributes=[tgt], feature_attributes=["feature"])
    ie = get_inference_engine("sklearn", config=config)
    assert isinstance(ie, SklearnInferenceEngine)
    ie.load_and_split_data(collection)
    ie.initialize_model()
    assert len(ie.transformed_features) == 5, "expected encoding of categories"
    assert ie.encoders
    outcome = check_accuracy2(ie, tgt, threshold=0.0, test_data=pd.DataFrame(test_data))
    assert outcome.accuracy == 0.0


def test_multivalued():
    client = Client()
    n = 100
    tgt = "code"
    features = ["preceding"]
    letters = list("abcdefghijklmnopqrstuvwxyz")

    def _code(i: int) -> str:
        return letters[i % 10]

    def _preceding(i: int) -> List[str]:
        codes = list(letters[0 : i % 10])
        pos = randint(0, 10)
        if pos < len(codes):
            codes.pop(pos)
        return codes

    objects = [{"code": _code(i), "preceding": _preceding(i)} for i in range(n)]
    db = client.attach_database("duckdb", alias="test")
    db.store({"data": objects})
    collection = db.get_collection("data")
    config = InferenceConfig(target_attributes=[tgt], feature_attributes=features)
    ie = get_inference_engine("sklearn", config=config)
    ie.load_and_split_data(collection)
    ie.initialize_model()
    check_accuracy(ie, tgt, threshold=0.5, test_data=ie.testing_data.as_dataframe())
    assert isinstance(ie, SklearnInferenceEngine)
    # io = StringIO()
    # ie.export_model(io, model_serialization=ModelSerialization.LINKML_EXPRESSION)
    # rule_engine = get_inference_engine('rulebased')
    # rule_engine.import_model_from(ie)


@pytest.mark.parametrize(
    "prop_age_missing,prop_stage_missing",
    [
        (0.0, 0.0),
        (0.05, 0.0),
        (0.05, 0.05),
        (0.0, 0.05),
        (0.05, 0.05),
    ],
)
def test_missing(prop_age_missing: float, prop_stage_missing: float):
    client = Client()

    STAGE_MAP = {
        "childhood": (1, 5),
        "juvenile": (5, 15),
        "adult": (16, None),
    }

    n = 1000
    max_age = 80
    tgt = "stage"

    def _obj(i: int) -> dict:
        age = (i % (max_age - 1)) + 1
        for stage, (low, high) in STAGE_MAP.items():
            if high is None or low <= age < high:
                break
        if prop_age_missing > 0 and random.random() < prop_age_missing:
            age = None
        if prop_stage_missing > 0 and random.random() < prop_stage_missing:
            stage = None
        return {"id": i, "age": age, "stage": stage}

    objects = [_obj(i) for i in range(n)]
    print(objects)
    db = client.attach_database("duckdb", alias="test")
    db.store({"data": objects})
    collection = db.get_collection("data")
    config = InferenceConfig(target_attributes=[tgt])
    ie = get_inference_engine("sklearn", config=config)
    ie.load_and_split_data(collection)
    ie.initialize_model()
    # crude heuristic - in theory this could fail spuriously
    threshold = 0.7 - (prop_age_missing + prop_stage_missing)
    check_accuracy(ie, tgt, threshold=threshold, test_data=ie.testing_data.as_dataframe())
    assert isinstance(ie, SklearnInferenceEngine)
    ie.export_model(
        OUTPUT_DIR / f"test_missing_{prop_age_missing}-{prop_stage_missing}.png",
        model_serialization=ModelSerialization.PNG,
    )
    io = StringIO()
    ie.export_model(io, model_serialization=ModelSerialization.LINKML_EXPRESSION)
    logger.info(f"RULES: {io.getvalue()}")
    rule_engine = get_inference_engine("rulebased")
    rule_engine.import_model_from(ie)
    if True or not prop_stage_missing and not prop_age_missing:
        check_accuracy(rule_engine, tgt, threshold=threshold, test_data=ie.testing_data.as_dataframe())
