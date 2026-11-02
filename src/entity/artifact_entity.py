from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    raw_data_file_path: str
    



@dataclass
class DataValidationArtifact:

    validation_status: bool

    validation_report_file_path: str
    

@dataclass
class DataTransformationArtifact:
    train_features_path: str
    test_features_path: str

    train_target_path: str
    test_target_path: str

    preprocessor_file_path: str
    



@dataclass
class ModelTrainerArtifact:
    logistic_regression_model_path: str
    decision_tree_model_path: str
    random_forest_model_path: str
    
    
    
@dataclass
class ModelEvaluationArtifact:
    model_evaluation_report_path: str
    best_model_name: str
    best_model_score: float
    
    
@dataclass
class ModelPusherArtifact:
    pushed_model_path: str
    pushed_preprocessor_path: str