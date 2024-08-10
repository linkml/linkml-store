from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union, TextIO

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from linkml_store.api.collection import OBJECT, Collection
from linkml_store.inference.inference_config import InferenceConfig, Inference


class ModelSerialization(str, Enum):
    """
    Enum for model serialization types.
    """
    PICKLE = "pickle"
    ONNX = "onnx"
    PMML = "pmml"
    PFA = "pfa"
    LINKML_EXPRESSION = "linkml_expression"


class CollectionSlice(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    collection: Optional[Collection] = None
    dataframe: Optional[pd.DataFrame] = None
    slice: Tuple[Optional[int], Optional[int]] = Field(default=(None, None))

    def as_dataframe(self) -> pd.DataFrame:
        """
        Return the slice of the collection as a dataframe.

        :return:
        """
        if self.dataframe is not None:
            df = self.dataframe
            return df.iloc[self.slice[0] : self.slice[1]]
        elif self.collection is not None:
            rs = self.collection.find({}, offset=self.slice[0], limit=self.slice[1] - self.slice[0])
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

    def load_and_split_data(self, collection: Collection, split: Tuple[float, float] = (0.7, 0.3)):
        """
        Load the data and split it into training and testing sets.

        :param collection:
        :param split:
        :return:
        """
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
            self,
            output: Optional[Union[str, Path, TextIO]],
            model_serialization: ModelSerialization,
             **kwargs):
        """
        Export the model to the given output.

        :param model_serialization:
        :param output:
        :param kwargs:
        :return:
        """
        raise NotImplementedError("Export model method must be implemented by subclass")

    def derive(self, object: OBJECT) -> Optional[Inference]:
        """
        Derive the prediction for the given object.

        :param object:
        :return:
        """
        raise NotImplementedError("Predict method must be implemented by subclass")