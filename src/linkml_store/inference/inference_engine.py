import logging
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, TextIO, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from linkml_store.api.collection import OBJECT, Collection
from linkml_store.inference.inference_config import Inference, InferenceConfig
from linkml_store.utils.pandas_utils import nested_objects_to_dataframe

logger = logging.getLogger(__name__)


class ModelSerialization(str, Enum):
    """
    Enum for model serialization types.
    """

    PICKLE = "pickle"
    ONNX = "onnx"
    PMML = "pmml"
    PFA = "pfa"
    JOBLIB = "joblib"
    PNG = "png"
    LINKML_EXPRESSION = "linkml_expression"
    RULE_BASED = "rulebased"

    @classmethod
    def from_filepath(cls, file_path: str) -> Optional["ModelSerialization"]:
        """
        Get the serialization type from the file path.

        >>> ModelSerialization.from_filepath("model.onnx")
        <ModelSerialization.ONNX: 'onnx'>
        >>> ModelSerialization.from_filepath("model.pkl")
        <ModelSerialization.PICKLE: 'pickle'>
        >>> assert ModelSerialization.from_filepath("poor_file_name") is None

        :param file_path:
        :return:
        """
        toks = file_path.split(".")
        suffix = toks[-1]
        if len(toks) > 2:
            if suffix == "yaml" and toks[-2] == "rulebased":
                return cls.RULE_BASED
        # Generate mapping dynamically
        extension_mapping = {v.lower(): v for v in cls}
        # Add special cases
        extension_mapping["pkl"] = cls.PICKLE
        extension_mapping["py"] = cls.LINKML_EXPRESSION
        return extension_mapping.get(suffix, None)


class CollectionSlice(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    collection: Optional[Collection] = None
    dataframe: Optional[pd.DataFrame] = None
    slice: Tuple[Optional[int], Optional[int]] = Field(default=(None, None))

    def as_dataframe(self, flattened=False) -> pd.DataFrame:
        """
        Return the slice of the collection as a dataframe.

        :return:
        """
        if self.dataframe is not None:
            df = self.dataframe
            return df.iloc[self.slice[0] : self.slice[1]]
        elif self.collection is not None:
            rs = self.collection.find({}, offset=self.slice[0], limit=self.slice[1] - self.slice[0])
            if flattened:
                return nested_objects_to_dataframe(rs.rows)
            else:
                return rs.rows_dataframe
        else:
            raise ValueError("No dataframe or collection provided")


@dataclass
class InferenceEngine(ABC):
    """
    Base class for all inference engine.

    An InferenceEngine is capable of deriving inferences from input objects and a collection.
    """

    predictor_type: Optional[str] = None
    config: Optional[InferenceConfig] = None

    training_data: Optional[CollectionSlice] = None
    testing_data: Optional[CollectionSlice] = None

    def load_and_split_data(self, collection: Collection, split: Optional[Tuple[float, float]] = None):
        """
        Load the data and split it into training and testing sets.

        :param collection:
        :param split:
        :return:
        """
        split = split or self.config.train_test_split
        if not split:
            split = (0.7, 0.3)
        logger.info(f"Loading and splitting data from collection {collection.alias}")
        size = collection.size()
        self.training_data = CollectionSlice(collection=collection, slice=(0, int(size * split[0])))
        self.testing_data = CollectionSlice(collection=collection, slice=(int(size * split[0]), size))

    def initialize_model(self, **kwargs):
        """
        Initialize the model.

        :param kwargs:
        :return:
        """
        raise NotImplementedError("Initialize model method must be implemented by subclass")

    def export_model(
        self, output: Optional[Union[str, Path, TextIO]], model_serialization: ModelSerialization = None, **kwargs
    ):
        """
        Export the model to the given output.

        :param model_serialization:
        :param output:
        :param kwargs:
        :return:
        """
        raise NotImplementedError("Export model method must be implemented by subclass")

    def import_model_from(self, inference_engine: "InferenceEngine", **kwargs):
        """
        Import the model from the given inference engine.

        :param inference_engine:
        :param kwargs:
        :return:
        """
        raise NotImplementedError("Import model method must be implemented by subclass")

    def save_model(self, output: Union[str, Path]) -> None:
        """
        Save the model to the given output.

        :param output:
        :return:
        """
        raise NotImplementedError("Save model method must be implemented by subclass")

    @classmethod
    def load_model(cls, file_path: Union[str, Path]) -> "InferenceEngine":
        """
        Load the model from the given file path.

        :param file_path:
        :return:
        """
        raise NotImplementedError("Load model method must be implemented by subclass")

    def derive(self, object: OBJECT) -> Optional[Inference]:
        """
        Derive the prediction for the given object.

        :param object:
        :return:
        """
        raise NotImplementedError("Predict method must be implemented by subclass")
