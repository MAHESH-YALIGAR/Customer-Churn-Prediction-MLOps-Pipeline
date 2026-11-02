from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import Datavalidationconnfig
from src.entity.config_entity import DataTransformationConfig
from src.entity.config_entity import DataTransformationConfig
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

def get_evaluation_model_config()->ModelEvaluationConfig:
    return ModelEvaluationConfig()

def get_model_pusher_config()->ModelPusherConfig:
    return ModelPusherConfig()