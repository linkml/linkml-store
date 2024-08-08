import logging
from abc import ABC
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from linkml_store.api.collection import OBJECT
from linkml_store.predict.predictor_config import PredictorConfig, Prediction
from linkml_store.predict.predictor import Predictor

logger = logging.getLogger(__name__)

@dataclass
class SklearnPredictor(Predictor):
    """
    scikit-learn based predictor.


    >>> from linkml_store.api.client import Client
    >>> from linkml_store.utils.format_utils import Format
    >>> client = Client()
    >>> db = client.attach_database("duckdb", alias="test")
    >>> db.import_database("tests/input/iris.csv", Format.CSV, collection_name="iris")
    >>> db.list_collection_names()
    ['iris']
    >>> collection = db.get_collection("iris")
    >>> collection.find({}).num_rows
    150

    >>> features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    >>> config = PredictorConfig(target_attributes=["species"], feature_attributes=features)
    >>> predictor = SklearnPredictor(config=config)
    >>> predictor.load_and_split_data(collection)
    >>> predictor.initialize_model()
    >>> prediction = predictor.derive({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    >>> prediction.predicted_object
    {'species': 'setosa'}

    """
    classifier: Any = None
    encoders: dict = None

    def initialize_model(self, **kwargs):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import LabelEncoder

        encoders = {}

        df = self.training_data.as_dataframe()

        feature_cols = self.config.feature_attributes
        target_cols = self.config.target_attributes
        logger.info(f"Feature columns: {feature_cols}")
        logger.info(f"Target columns: {target_cols}")
        if len(target_cols) != 1:
            raise ValueError("Only one target column is supported")
        target_col = target_cols[0]

        # Prepare the data
        X = df[feature_cols].copy()  # Create an explicit copy
        y = df[target_col].copy()

        # Encode categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                encoders[col] = LabelEncoder()
                X[col] = encoders[col].fit_transform(X[col].astype(str))

        if y.dtype == 'object':
            y_encoder = LabelEncoder()
            encoders[target_col] = y_encoder
            y = y_encoder.fit_transform(y.astype(str))

        # Train a decision tree and get cross-validated accuracy
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        self.classifier = clf
        self.encoders = encoders

    def derive(self, object: OBJECT) -> Optional[Prediction]:
        from sklearn.tree import DecisionTreeClassifier
        encoders = self.encoders
        target_attributes = self.config.target_attributes
        target_attribute = target_attributes[0]
        y_encoder = encoders.get(target_attribute)

        clf: DecisionTreeClassifier = self.classifier
        new_X = pd.DataFrame([object])
        for col in new_X.columns:
            if col in encoders:
                new_X[col] = encoders[col].transform(new_X[col].astype(str))
        predictions = clf.predict(new_X)
        if y_encoder:
            v = y_encoder.inverse_transform(predictions)
        else:
            v = predictions
        predicted_object = {target_attribute: v[0]}
        logger.info(f"Predicted object: {predicted_object}")
        return Prediction(predicted_object=predicted_object)

