import logging
import random
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TextIO, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict

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
    RAG_INDEX = "rag_index"

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
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: Optional[str] = None
    base_collection: Optional[Collection] = None
    # _dataframe: Optional[pd.DataFrame] = None
    # slice: Tuple[Optional[int], Optional[int]] = Field(default=(None, None))
    indices: Optional[Tuple[int, ...]] = None
    _collection: Optional[Collection] = None
    where: Any = None

    @property
    def collection(self) -> Collection:
        if not self._collection and not self.indices:
            return self.base_collection
        if not self._collection:
            rows = self.base_collection.rows
            subset = [rows[i] for i in self.indices]
            db = self.base_collection.parent
            subset_name = self.slice_alias
            subset_collection = db.get_collection(subset_name, create_if_not_exists=True)
            # ensure the collection has the same schema type as the base collection;
            # this ensures that column/attribute types are preserved
            subset_collection.metadata.type = self.base_collection.target_class_name
            subset_collection.delete_where({})
            subset_collection.insert(subset)
            self._collection = subset_collection
        return self._collection

    @property
    def slice_alias(self) -> str:
        return f"{self.base_collection.alias}__rag_{self.name}"

    def as_dataframe(self, flattened=False) -> pd.DataFrame:
        """
        Return the slice of the collection as a dataframe.

        :param flattened: flattned nested objects to give keys like foo.bar
        :return:
        """
        rs = self.collection.find({}, limit=-1)
        if flattened:
            return nested_objects_to_dataframe(rs.rows)
        else:
            return rs.rows_dataframe


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

    def load_and_split_data(self, collection: Collection, split: Optional[Tuple[float, float]] = None, randomize=True):
        """
        Load the data and split it into training and testing sets.

        :param collection:
        :param split: Tuple of training and testing split ratios.
        :param randomize:
        :return:
        """
        local_random = random.Random(self.config.random_seed) if self.config.random_seed else random.Random()
        split = split or self.config.train_test_split
        if not split:
            split = (0.7, 0.3)
        if split[0] == 1.0:
            self.training_data = CollectionSlice(name="train", base_collection=collection, indices=None)
            self.testing_data = None
            return
        logger.info(f"Loading and splitting data {split} from collection {collection.alias}")
        size = collection.size()
        indices = range(size)
        if randomize:
            train_indices = local_random.sample(indices, int(size * split[0]))
            test_indices = set(indices) - set(train_indices)
        else:
            train_indices = indices[: int(size * split[0])]
            test_indices = indices[int(size * split[0]) :]
        self.training_data = CollectionSlice(name="train", base_collection=collection, indices=train_indices)
        self.testing_data = CollectionSlice(name="test", base_collection=collection, indices=test_indices)

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
