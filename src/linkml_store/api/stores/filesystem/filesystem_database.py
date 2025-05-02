import logging
from pathlib import Path
from typing import Optional

import yaml
from linkml_runtime import SchemaView
from linkml_runtime.utils.schema_builder import SchemaBuilder

from linkml_store.api import Database
from linkml_store.api.config import DatabaseConfig
from linkml_store.api.stores.filesystem.filesystem_collection import FileSystemCollection
from linkml_store.utils.file_utils import safe_remove_directory
from linkml_store.utils.format_utils import Format, load_objects

logger = logging.getLogger(__name__)


class FileSystemDatabase(Database):
    collection_class = FileSystemCollection

    directory_path: Optional[Path] = None
    default_file_format: Optional[str] = None

    no_backup_on_drop: bool = False

    def __init__(self, handle: Optional[str] = None, **kwargs):
        handle = handle.replace("file:", "")
        if handle.startswith("//"):
            handle = handle[2:]
        self.directory_path = Path(handle)
        self.load_metadata()
        super().__init__(handle=handle, **kwargs)

    @property
    def metadata_path(self) -> Path:
        return self.directory_path / ".linkml_metadata.yaml"

    def load_metadata(self):
        if self.metadata_path.exists():
            md_dict = yaml.safe_load(open(self.metadata_path))
            metadata = DatabaseConfig(**md_dict)
        else:
            metadata = DatabaseConfig()
        self.metadata = metadata

    def close(self, **kwargs):
        pass

    def drop(self, no_backup=False, **kwargs):
        self.close()
        path = self.directory_path
        if path.exists():
            safe_remove_directory(path, no_backup=self.no_backup_on_drop or no_backup)

    def init_collections(self):
        metadata = self.metadata
        if self._collections is None:
            self._collections = {}
        for name, collection_config in metadata.collections.items():
            collection = FileSystemCollection(parent=self, **collection_config.dict())
            self._collections[name] = collection
        path = self.directory_path
        if path.exists():
            for fmt in Format:
                suffix = fmt.value
                logger.info(f"Looking for {suffix} files in {path}")
                for f in path.glob(f"*.{suffix}"):
                    logger.info(f"Found {f}")
                    n = f.stem
                    objs = load_objects(f, suffix, expected_type=list)
                    collection = FileSystemCollection(parent=self, name=n)
                    self._collections[n] = collection
                    collection._set_objects(objs)

    def xxxinduce_schema_view(self) -> SchemaView:
        logger.info(f"Inducing schema view for {self.handle}")
        sb = SchemaBuilder()

        for collection_name in self.list_collection_names():
            sb.add_class(collection_name)
        return SchemaView(sb.schema)
