import json

from src.constants import TRAINED_MODEL_DIR
from src.entity.artifact_entity import ModelTrainerArtifact
from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import Datavalidationconnfig
from src.entity.config_entity import DataTransformationConfig
from src.entity.config_entity import DataTransformationConfig
from src.entity.config_entity import ModelTrainerConfig
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.config_entity import ModelPusherConfig

def get_data_ingestion_config() -> DataIngestionConfig:
    return DataIngestionConfig()

def  get_data_validation_config()->Datavalidationconnfig:
    return Datavalidationconnfig()

def get_data_transformation_config()->DataTransformationConfig:
    return DataTransformationConfig()


def get_model_training()->DataTransformationConfig:
    return DataTransformationConfig()

def get_model_trainer_artifact() -> ModelTrainerArtifact:
    config = ModelTrainerConfig(trained_model_dir=TRAINED_MODEL_DIR)

    with open(config.run_ids_path, encoding="utf-8") as file:
        run_ids = json.load(file)

    return ModelTrainerArtifact(
        logistic_regression_model_path=config.logistic_regression_model_path,
        decision_tree_model_path=config.decision_tree_model_path,
        random_forest_model_path=config.random_forest_model_path,
        logistic_regression_run_id=run_ids["logistic_regression"],
        decision_tree_run_id=run_ids["decision_tree"],
        random_forest_run_id=run_ids["random_forest"],
    )

def get_evaluation_model_config()->ModelEvaluationConfig:
    return ModelEvaluationConfig()

def get_model_pusher_config()->ModelPusherConfig:
    return ModelPusherConfig()