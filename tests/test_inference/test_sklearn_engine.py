import json
from io import StringIO
from typing import Optional, Tuple, List

import numpy as np

from linkml_store.api.client import Client
from linkml_store.inference import InferenceConfig, get_inference_engine, SklearnInferenceEngine
from linkml_store.inference.inference_engine import ModelSerialization
from linkml_store.utils.format_utils import Format
from tests import INPUT_DIR
import pandas as pd
from linkml_runtime.utils.eval_utils import eval_expr


def get_dataset(name: str, version: Optional[int] = 2) -> Tuple[pd.DataFrame, List[str], List[str]]:
    from sklearn.datasets import fetch_openml
    dataset = fetch_openml(name=name, version=version, as_frame=True)
    df = pd.concat([dataset.data, dataset.target], axis=1)
    return df, list(dataset.data.columns), ['class']


def test_inference_basic():
    """
    Test the sklearn inference engine
    :return:
    """
    client = Client()
    db = client.attach_database("duckdb", alias="test")
    db.import_database(INPUT_DIR / "iris.jsonl", Format.JSONL, collection_name="iris")
    assert db.list_collection_names() == ['iris']
    collection = db.get_collection("iris")
    assert collection.find({}).num_rows == 100  # TODO
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    config = InferenceConfig(target_attributes=["species"], feature_attributes=features)
    ie = get_inference_engine('sklearn', config=config)
    ie.load_and_split_data(collection)
    ie.initialize_model()
    assert isinstance(ie, SklearnInferenceEngine)
    assert set(ie.encoders.keys()) == {"species"}
    q = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    prediction = ie.derive(q)
    assert prediction.predicted_object['species'] == 'setosa'
    io = StringIO()
    ie.export_model(io, model_serialization=ModelSerialization.LINKML_EXPRESSION)
    expr = io.getvalue()
    print(expr)
    assert eval_expr(expr, **q) == 'setosa'


def test_inference_mixed():
    """
    Test the sklearn inference engine
    :return:
    """
    client = Client()
    df, features, targets = get_dataset("adult", 2)
    print(f"Features: {type(features)} {type(features[0])} {features}")
    print(df.columns)
    print(df)
    df = df.replace({np.nan: None})
    # https://github.com/pandas-dev/pandas/issues/58230
    rows = json.loads(df.to_json(orient='records'))
    db = client.attach_database("duckdb", alias="test")
    db.store({"data": rows})
    collection = db.get_collection("data")
    config = InferenceConfig(target_attributes=targets, feature_attributes=features)
    ie = get_inference_engine('sklearn', config=config)
    ie.load_and_split_data(collection)
    ie.initialize_model()
    n = 0
    tp = 0
    for test_row in ie.testing_data.as_dataframe().to_dict(orient='records')[0:10]:
        expected = test_row.pop('class')
        prediction = ie.derive(test_row)
        if prediction.predicted_object['class'] == expected:
            tp += 1
        n += 1
    print(f"Accuracy: {tp/n}")


def test_nested_data():
    """
    Test the sklearn inference engine with nested data

    :return:
    """
    client = Client()
    n = 100
    def _age(i: int) -> int:
        return i % 10 + 20

    objects = [
        {"person": { "name": f"Person {i}", "age": _age(i) } ,
         "twin": {"name": f"Twin {i}", "age": _age(i)}}
        for i in range(n)
    ]
    db = client.attach_database("duckdb", alias="test")
    db.store({"data": objects})
    collection = db.get_collection("data")
    config = InferenceConfig(target_attributes=["twin.age"], feature_attributes=["person.name", "person.age"])
    ie = get_inference_engine('sklearn', config=config)
    ie.load_and_split_data(collection)
    ie.initialize_model()
    n = 0
    tp = 0
    for test_row in ie.testing_data.as_dataframe().to_dict(orient='records')[0:10]:
        expected = test_row.pop('class')
        prediction = ie.derive(test_row)
        if prediction.predicted_object['class'] == expected:
            tp += 1
        n += 1
    print(f"Accuracy: {tp/n}")