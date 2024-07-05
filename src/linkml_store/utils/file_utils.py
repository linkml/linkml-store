import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

# Set up logging
logger = logging.getLogger(__name__)


def safe_remove_directory(dir_path: Path, no_backup: bool = False) -> Optional[Path]:
    # Ensure the directory exists
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {dir_path}")
    try:
        if no_backup:
            # Move to a temporary directory instead of permanent removal
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / dir_path.name
                shutil.move(str(dir_path), str(tmp_path))
                logger.info(f"Directory moved to temporary location: {tmp_path}")
                # The directory will be automatically removed when exiting the context manager
            return None
        else:
            # Create a backup directory name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = dir_path.with_name(f"{dir_path.name}_backup_{timestamp}")

            # Move the directory to the backup location
            shutil.move(str(dir_path), str(backup_dir))
            logger.info(f"Directory backed up to: {backup_dir}")
            return backup_dir

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None
