import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, TextIO, Type, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from linkml_store.api.collection import OBJECT
from linkml_store.inference.implementations.rule_based_inference_engine import RuleBasedInferenceEngine
from linkml_store.inference.inference_config import Inference, InferenceConfig
from linkml_store.inference.inference_engine import InferenceEngine, ModelSerialization
from linkml_store.utils.sklearn_utils import tree_to_nested_expression, visualize_decision_tree

logger = logging.getLogger(__name__)


@dataclass
class SklearnInferenceEngine(InferenceEngine):
    config: InferenceConfig
    classifier: Any = None
    encoders: Dict[str, Any] = field(default_factory=dict)
    transformed_features: List[str] = field(default_factory=list)
    transformed_targets: List[str] = field(default_factory=list)
    skip_features: List[str] = field(default_factory=list)
    categorical_encoder_class: Optional[Type[Union[OneHotEncoder, MultiLabelBinarizer]]] = None
    maximum_proportion_distinct_features: float = 0.2
    confidence: float = 0.0

    strict: bool = False

    PERSIST_COLS: ClassVar = [
        "config",
        "classifier",
        "encoders",
        "transformed_features",
        "transformed_targets",
        "skip_features",
        "confidence",
    ]

    def _get_encoder(self, v: Union[List[Any], Any]) -> Any:
        if isinstance(v, list):
            if all(isinstance(x, list) for x in v):
                return MultiLabelBinarizer()
            elif all(isinstance(x, str) for x in v):
                return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            elif all(isinstance(x, (int, float)) for x in v):
                return None
            else:
                raise ValueError("Mixed data types in the list are not supported")
        else:
            if hasattr(v, "dtype"):
                if v.dtype == "object" or v.dtype.name == "category":
                    if isinstance(v.iloc[0], list):
                        return MultiLabelBinarizer()
                    elif self.categorical_encoder_class:
                        return self.categorical_encoder_class(handle_unknown="ignore")
                    else:
                        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                elif v.dtype.kind in "biufc":
                    return None
        raise ValueError("Unable to determine appropriate encoder for the input data")

    def _is_complex_column(self, column: pd.Series) -> bool:
        """Check if the column contains complex data types like lists or dicts."""
        # MV_TYPE = (list, dict)
        MV_TYPE = (list,)
        return (column.dtype == "object" or column.dtype == "category") and any(
            isinstance(x, MV_TYPE) for x in column.dropna()
        )

    def _get_unique_values(self, column: pd.Series) -> set:
        """Get unique values from a column, handling list-type data."""
        if self._is_complex_column(column):
            # For columns with lists, flatten the lists and get unique values
            return set(
                item for sublist in column.dropna() for item in (sublist if isinstance(sublist, list) else [sublist])
            )
        else:
            return set(column.unique())

    def initialize_model(self, **kwargs):
        logger.info(f"Initializing model with config: {self.config}")
        df = self.training_data.as_dataframe(flattened=True)
        logger.info(f"Training data shape: {df.shape}")
        target_cols = self.config.target_attributes
        feature_cols = self.config.feature_attributes
        if len(target_cols) != 1:
            raise ValueError("Only one target column is supported")
        if not feature_cols:
            feature_cols = df.columns.difference(target_cols).tolist()
            self.config.feature_attributes = feature_cols
            if not feature_cols:
                raise ValueError("No features found in the data")
        target_col = target_cols[0]
        logger.info(f"Feature columns: {feature_cols}")
        X = df[feature_cols].copy()
        logger.info(f"Target column: {target_col}")
        y = df[target_col].copy()

        # find list of features to skip (categorical with > N categories)
        skip_features = []
        if not len(X.columns):
            raise ValueError("No features to train on")
        for col in X.columns:
            unique_values = self._get_unique_values(X[col])
            if len(unique_values) > self.maximum_proportion_distinct_features * len(X[col]):
                skip_features.append(col)
            if False and (X[col].dtype == "object" or X[col].dtype.name == "category"):
                if len(X[col].unique()) > self.maximum_proportion_distinct_features * len(X[col]):
                    skip_features.append(col)
        self.skip_features = skip_features
        X = X.drop(skip_features, axis=1)
        logger.info(f"Skipping features: {skip_features}")

        # Encode features
        encoded_features = []
        if not len(X.columns):
            raise ValueError(f"No features to train on from after skipping {skip_features}")
        for col in X.columns:
            logger.info(f"Checking whether to encode: {col}")
            col_encoder = self._get_encoder(X[col])
            if col_encoder:
                self.encoders[col] = col_encoder
                if isinstance(col_encoder, OneHotEncoder):
                    encoded = col_encoder.fit_transform(X[[col]])
                    feature_names = col_encoder.get_feature_names_out([col])
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
                    X = pd.concat([X.drop(col, axis=1), encoded_df], axis=1)
                    encoded_features.extend(feature_names)
                elif isinstance(col_encoder, MultiLabelBinarizer):
                    encoded = col_encoder.fit_transform(X[col])
                    feature_names = [f"{col}_{c}" for c in col_encoder.classes_]
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
                    X = pd.concat([X.drop(col, axis=1), encoded_df], axis=1)
                    encoded_features.extend(feature_names)
                else:
                    X[col] = col_encoder.fit_transform(X[col])
                    encoded_features.append(col)
            else:
                encoded_features.append(col)

        self.transformed_features = encoded_features
        logger.info(f"Encoded features: {self.transformed_features}")
        logger.info(f"Number of features after encoding: {len(self.transformed_features)}")

        # Encode target
        # y_encoder = LabelEncoder()
        y_encoder = self._get_encoder(y)
        if isinstance(y_encoder, OneHotEncoder):
            y_encoder = LabelEncoder()
        # self.encoders[target_col] = y_encoder
        if y_encoder:
            self.encoders[target_col] = y_encoder
            y = y_encoder.fit_transform(y.values.ravel())  # Convert to 1D numpy array
            self.transformed_targets = y_encoder.classes_

        # print(f"Fitting model with features: {X.columns}, y={y}, X={X}")
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        self.classifier = clf
        logger.info("Model fit complete")
        cv_scores = cross_val_score(self.classifier, X, y, cv=5)
        self.confidence = cv_scores.mean()
        logger.info(f"Cross-validation scores: {cv_scores}")

    def derive(self, object: OBJECT) -> Optional[Inference]:
        object = self._normalize(object)
        new_X = pd.DataFrame([object])

        # Apply encodings
        encoded_features = {}
        for col in self.config.feature_attributes:
            if col in self.skip_features:
                continue
            if col in self.encoders:
                encoder = self.encoders[col]
                if isinstance(encoder, OneHotEncoder):
                    print(f"Encoding: {col} v={object[col]} df={new_X[[col]]} encoder={encoder}")
                    encoded = encoder.transform(new_X[[col]])
                    feature_names = encoder.get_feature_names_out([col])
                    for i, name in enumerate(feature_names):
                        encoded_features[name] = encoded[0, i]
                elif isinstance(encoder, MultiLabelBinarizer):
                    encoded = encoder.transform(new_X[col])
                    feature_names = [f"{col}_{c}" for c in encoder.classes_]
                    for i, name in enumerate(feature_names):
                        encoded_features[name] = encoded[0, i]
                else:  # LabelEncoder or similar
                    encoded_features[col] = encoder.transform(new_X[col].astype(str))[0]
            else:
                encoded_features[col] = new_X[col].iloc[0]

        # Ensure all expected features are present and in the correct order
        final_features = []
        for feature in self.transformed_features:
            if feature in encoded_features:
                final_features.append(encoded_features[feature])
            else:
                final_features.append(0)  # or some other default value

        # Create the final input array
        new_X_array = np.array(final_features).reshape(1, -1)

        logger.info(f"Input features: {self.transformed_features}")
        logger.info(f"Number of input features: {len(self.transformed_features)}")

        predictions = self.classifier.predict(new_X_array)
        target_attribute = self.config.target_attributes[0]
        y_encoder = self.encoders.get(target_attribute)

        if y_encoder:
            v = y_encoder.inverse_transform(predictions)
        else:
            v = predictions

        predicted_object = {target_attribute: v[0]}
        logger.info(f"Predicted object: {predicted_object}")
        return Inference(predicted_object=predicted_object, confidence=self.confidence)

    def _normalize(self, object: OBJECT) -> OBJECT:
        """
        Normalize the input object to ensure it has all the expected attributes.

        Also remove any numpy/pandas oddities

        :param object:
        :return:
        """
        np_map = {np.nan: None}

        def _tr(x: Any):
            # TODO: figure a more elegant way to do this
            try:
                return np_map.get(x, x)
            except TypeError:
                return x

        return {k: _tr(object.get(k, None)) for k in self.config.feature_attributes}

    def export_model(
        self, output: Optional[Union[str, Path, TextIO]], model_serialization: ModelSerialization = None, **kwargs
    ):
        def as_file():
            if isinstance(output, (str, Path)):
                return open(output, "w")
            return output

        if model_serialization is None:
            if isinstance(output, (str, Path)):
                model_serialization = ModelSerialization.from_filepath(output)
            if model_serialization is None:
                model_serialization = ModelSerialization.JOBLIB

        if model_serialization == ModelSerialization.LINKML_EXPRESSION:
            expr = tree_to_nested_expression(
                self.classifier,
                self.transformed_features,
                self.encoders.keys(),
                feature_encoders=self.encoders,
                target_encoder=self.encoders.get(self.config.target_attributes[0]),
            )
            as_file().write(expr)
        elif model_serialization == ModelSerialization.JOBLIB:
            self.save_model(output)
        elif model_serialization == ModelSerialization.RULE_BASED:
            rbie = RuleBasedInferenceEngine(config=self.config)
            rbie.import_model_from(self)
            rbie.save_model(output)
        elif model_serialization == ModelSerialization.PNG:
            visualize_decision_tree(self.classifier, self.transformed_features, self.transformed_targets, output)
        else:
            raise ValueError(f"Unsupported model serialization: {model_serialization}")

    def save_model(self, output: Union[str, Path]) -> None:
        """
        Save the trained model and related data to a file.

        :param output: Path to save the model
        """
        import joblib

        if self.classifier is None:
            raise ValueError("Model has not been trained. Call initialize_model() first.")

        # Use self.PERSIST_COLS
        model_data = {k: getattr(self, k) for k in self.PERSIST_COLS}

        joblib.dump(model_data, output)

    @classmethod
    def load_model(cls, file_path: Union[str, Path]) -> "SklearnInferenceEngine":
        """
        Load a trained model and related data from a file.

        :param file_path: Path to the saved model
        :return: SklearnInferenceEngine instance with loaded model
        """
        import joblib

        model_data = joblib.load(file_path)

        engine = cls(config=model_data["config"])
        for k, v in model_data.items():
            if k == "config":
                continue
            setattr(engine, k, v)

        logger.info(f"Model loaded from {file_path}")
        return engine
