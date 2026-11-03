from dataclasses import dataclass,field
from datetime import datetime

from src.constants import RAW_DATA_DIR, RAW_DATA_FILE_PATH, SOURCE_URL,REQUIRED_COLUMNS,REPORT_FILE_PATH,TRANSFORMED_DATA_DIR,TRAIN_FEATURES_PATH,TEST_FEATURES_PATH,TRAIN_TARGET_PATH,TEST_TARGET_PATH,PREPROCESSOR_FILE_PATH,TEST_SIZE,RANDOM_STATE,LOGISTIC_REGRESSION_MODEL_PATH,DECISION_TREE_MODEL_PATH,RANDOM_FOREST_MODEL_PATH,MODEL_RUN_IDS_PATH,MODEL_PUSHER_DIR,PUSHED_MODEL_PATH,PUSHED_PREPROCESSOR_PATH,MODEL_EVALUATION_REPORT_PATH,MODEL_EVALUATION_DIR

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class DataIngestionConfig:

    source_url: str = SOURCE_URL
    raw_data_dir: str = RAW_DATA_DIR
    raw_data_file_path: str = RAW_DATA_FILE_PATH

@dataclass
class Datavalidationconnfig:
    # required_columns: list=REQUIRED_COLUMNS
    required_columns: list = field(default_factory=lambda: REQUIRED_COLUMNS.copy())
    report_file_path: str=REPORT_FILE_PATH



from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    transformed_data_dir: str=TRANSFORMED_DATA_DIR

    train_features_path: str=TRAIN_FEATURES_PATH
    test_features_path: str=TEST_FEATURES_PATH

    train_target_path: str=TRAIN_TARGET_PATH
    test_target_path: str=TEST_TARGET_PATH

    preprocessor_file_path: str=PREPROCESSOR_FILE_PATH

    test_size: float=TEST_SIZE
    random_state: int=RANDOM_STATE
    
    
    
    


@dataclass
class ModelTrainerConfig:
    trained_model_dir: str

    logistic_regression_model_path: str=LOGISTIC_REGRESSION_MODEL_PATH
    decision_tree_model_path: str=DECISION_TREE_MODEL_PATH
    random_forest_model_path: str=RANDOM_FOREST_MODEL_PATH
    run_ids_path: str=MODEL_RUN_IDS_PATH
    
    
    
    
    
@dataclass
class ModelEvaluationConfig:
    model_evaluation_dir: str=MODEL_EVALUATION_DIR
    model_evaluation_report_path: str=MODEL_EVALUATION_REPORT_PATH
    
    


@dataclass
class ModelPusherConfig:
    model_pusher_dir: str=MODEL_PUSHER_DIR
    pushed_model_path: str=PUSHED_MODEL_PATH
    pushed_preprocessor_path: str=PUSHED_PREPROCESSOR_PATH
    