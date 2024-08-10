import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict, Union, TextIO

import pandas as pd

from linkml_store.api.collection import OBJECT
from linkml_store.inference.inference_config import InferenceConfig, Inference
from linkml_store.inference.inference_engine import InferenceEngine, ModelSerialization
from linkml_store.utils.sklearn_utils import tree_to_nested_expression

logger = logging.getLogger(__name__)


@dataclass
class SklearnInferenceEngine(InferenceEngine):
    """
    scikit-learn based inference engine.


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
    >>> config = InferenceConfig(target_attributes=["species"], feature_attributes=features)
    >>> ie = SklearnInferenceEngine(config=config)
    >>> ie.load_and_split_data(collection)
    >>> ie.initialize_model()
    >>> prediction = ie.derive({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    >>> prediction.predicted_object
    {'species': 'setosa'}

    """
    classifier: Any = None
    encoders: Optional[Dict[str, "sklearn.preprocessing.LabelEncoder"]] = None
    strict: bool = None

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

    def _normalize(self, object: OBJECT) -> OBJECT:
        """
        Normalize the object to match the feature attributes.

        scikit-learn requires that the input object has the same columns as the training data.

        :param object:
        :return:
        """
        object = {k: object.get(k, None) for k in self.config.feature_attributes}
        return object


    def derive(self, object: OBJECT) -> Optional[Inference]:
        object = self._normalize(object)
        from sklearn.tree import DecisionTreeClassifier
        encoders = self.encoders
        target_attributes = self.config.target_attributes
        target_attribute = target_attributes[0]
        y_encoder = encoders.get(target_attribute)

        clf: DecisionTreeClassifier = self.classifier
        new_X = pd.DataFrame([object])
        for col in new_X.columns:
            if col in encoders:
                try:
                    encoded = encoders[col].transform(new_X[col].astype(str))
                except ValueError as e:
                    logger.warning(f"Failed to encode column {col} not seen in training data")
                    encoded = None
                    if self.strict:
                        raise e
                new_X[col] = encoded
        predictions = clf.predict(new_X)
        if y_encoder:
            v = y_encoder.inverse_transform(predictions)
        else:
            v = predictions
        predicted_object = {target_attribute: v[0]}
        logger.info(f"Predicted object: {predicted_object}")
        return Inference(predicted_object=predicted_object)

    def export_model(self,
                     output: Optional[Union[str, Path, TextIO]],
                     model_serialization: ModelSerialization,
                     **kwargs):
        def as_file():
            if isinstance(output, (str, Path)):
                return open(output, 'w')
            return output
        if model_serialization == ModelSerialization.LINKML_EXPRESSION:
            expr = tree_to_nested_expression(self.classifier, self.config.feature_attributes, self.encoders.keys(),
                                             feature_encoders=self.encoders,
                                             target_encoder=self.encoders.get(self.config.target_attributes[0]))
            as_file().write(expr)
        else:
            raise ValueError(f"Unsupported model serialization: {model_serialization}")

