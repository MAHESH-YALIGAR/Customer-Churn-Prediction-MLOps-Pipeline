import sys

from src.Components.model_evaluation import ModelEvaluation
from src.Components.model_pusher import ModelPusher
from src.constants import (
    MODEL_EVALUATION_DIR,
    MODEL_EVALUATION_REPORT_PATH,
    TEST_FEATURES_PATH,
    TEST_TARGET_PATH,
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    PREPROCESSOR_FILE_PATH,
)

from src.entity.artifact_entity import (
    DataTransformationArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig
from src.exception import MyException
from src.logger import logging
from src.utils import get_model_pusher_config, get_model_trainer_artifact


def main() -> None:
    try:
        transformation_artifact = DataTransformationArtifact(
            train_features_path=TRAIN_FEATURES_PATH,
            test_features_path=TEST_FEATURES_PATH,
            train_target_path=TRAIN_TARGET_PATH,
            test_target_path=TEST_TARGET_PATH,
            preprocessor_file_path=PREPROCESSOR_FILE_PATH,
        )
        trainer_artifact = get_model_trainer_artifact()
        evaluation_config = ModelEvaluationConfig(
            model_evaluation_dir=MODEL_EVALUATION_DIR,
            model_evaluation_report_path=MODEL_EVALUATION_REPORT_PATH,
        )

        model_evaluation = ModelEvaluation(
            data_transformation_artifact=transformation_artifact,
            model_trainer_artifact=trainer_artifact,
            model_evaluation_config=evaluation_config,
        )
        evaluation_artifact = model_evaluation.initiate_model_evaluation()

        model_pusher = ModelPusher(
            data_transformation_artifact=transformation_artifact,
            model_trainer_artifact=trainer_artifact,
            model_evaluation_artifact=evaluation_artifact,
            model_pusher_config=get_model_pusher_config(),
        )
        model_pusher.initiate_model_pusher()

        logging.info("Best model: %s", evaluation_artifact.best_model_name)
        logging.info("Best model: %s", evaluation_artifact.best_model_name)
        logging.info(
            "Best model F1 score: %s",
            evaluation_artifact.best_model_score,
        )

    except Exception as e:
        logging.exception("Application failed")
        raise MyException(e, sys) from e


if __name__ == "__main__":
    main()
