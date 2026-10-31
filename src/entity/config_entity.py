from dataclasses import dataclass
from datetime import datetime

from src.constants import RAW_DATA_DIR, RAW_DATA_FILE_PATH, SOURCE_URL

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class DataIngestionConfig:

    source_url: str = SOURCE_URL
    raw_data_dir: str = RAW_DATA_DIR
    raw_data_file_path: str = RAW_DATA_FILE_PATH
