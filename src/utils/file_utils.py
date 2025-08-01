"""File management utilities."""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FileManager:
    """Manages file operations for the application."""
    
    def __init__(self, base_directory: Optional[str] = None):
        """Initialize file manager with optional base directory."""
        self.base_directory = Path(base_directory) if base_directory else Path.cwd()
        self.base_directory.mkdir(parents=True, exist_ok=True)
    
    def create_temp_file(self, suffix: str = ".pdf", prefix: str = "doc_") -> str:
        """Create a temporary file and return its path."""
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            prefix=prefix,
            dir=self.base_directory
        )
        temp_file.close()
        return temp_file.name
    
    def cleanup_temp_files(self, file_patterns: List[str]) -> None:
        """Clean up temporary files matching the given patterns."""
        for pattern in file_patterns:
            for file_path in self.base_directory.glob(pattern):
                try:
                    file_path.unlink()
                    logger.debug(f"Cleaned up temporary file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {file_path}: {e}")
    
    def ensure_directory(self, directory: str) -> Path:
        """Ensure directory exists and return Path object."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        return os.path.getsize(file_path)
    
    def file_exists(self, file_path: str) -> bool:
        """Check if file exists."""
        return Path(file_path).exists()

def create_temp_directory(prefix: str = "azure_search_") -> str:
    """Create a temporary directory and return its path."""
    return tempfile.mkdtemp(prefix=prefix) 