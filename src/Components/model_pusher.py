import os
import sys
import joblib

from src.logger import logging
from src.exception import MyException

from src.entity.config_entity import ModelPusherConfig

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)
import mlflow
from mlflow import MlflowClient

class ModelPusher:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig
    ):

        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

    def load_best_model(self):

        try:
            logging.info("Loading best trained model")

            best_model_name = (
                self.model_evaluation_artifact.best_model_name
            )

            # Map model names to their saved file paths
            model_paths = {
                "Logistic Regression":
                    self.model_trainer_artifact.logistic_regression_model_path,

                "Decision Tree":
                    self.model_trainer_artifact.decision_tree_model_path,

                "Random Forest":
                    self.model_trainer_artifact.random_forest_model_path
            }

            if best_model_name not in model_paths:
                raise ValueError(
                    f"Unknown best model: {best_model_name}"
                )

            best_model_path = model_paths[best_model_name]

            model = joblib.load(best_model_path)

            logging.info(
                f"Best model loaded successfully: {best_model_name}"
            )

            return model

        except Exception as e:
            raise MyException(e, sys)

    def push_model(self, model):

        try:
            logging.info("Pushing model to deployment directory")

            os.makedirs(
                self.model_pusher_config.model_pusher_dir,
                exist_ok=True
            )

            # Save selected model
            joblib.dump(
                model,
                self.model_pusher_config.pushed_model_path
            )

            # Load fitted preprocessor
            preprocessor = joblib.load(
                self.data_transformation_artifact.preprocessor_file_path
            )

            # Save preprocessor
            joblib.dump(
                preprocessor,
                self.model_pusher_config.pushed_preprocessor_path
            )

            logging.info("Model and preprocessor saved successfully")

        except Exception as e:
            raise MyException(e, sys)
    def register_model(self):

        try:
            logging.info("Starting MLflow model registration")

            mlflow.set_tracking_uri("http://127.0.0.1:5000")

            client = MlflowClient()

            best_model_name = (
                self.model_evaluation_artifact.best_model_name
            )

            run_ids = {
                "Logistic Regression":
                    self.model_trainer_artifact.logistic_regression_run_id,

                "Decision Tree":
                    self.model_trainer_artifact.decision_tree_run_id,

                "Random Forest":
                    self.model_trainer_artifact.random_forest_run_id
            }

            run_id = run_ids[best_model_name]

            model_uri = f"runs:/{run_id}/model"

            registered_model_name = "CustomerChurnModel"

            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name
            )

            logging.info(
                f"Model registered successfully. "
                f"Version: {model_version.version}"
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                alias="champion",
                version=model_version.version
            )

            logging.info("Champion alias assigned successfully")

        except Exception as e:
            raise MyException(e, sys)
    # def initiate_model_pusher(self):

    #     try:
    #         logging.info("Model pusher started")

    #         # Step 1: Load the best model
    #         best_model = self.load_best_model()

    #         # Step 2: Save model and preprocessor
    #         self.push_model(best_model)

    #         # Step 3: Return artifact
    #         model_pusher_artifact = ModelPusherArtifact(
    #             pushed_model_path=(
    #                 self.model_pusher_config.pushed_model_path
    #             ),
    #             pushed_preprocessor_path=(
    #                 self.model_pusher_config.pushed_preprocessor_path
    #             )
    #         )
    #         # logging.info("Model pusher completed in",self.pushed_model_path)

    #         logging.info("Model pusher completed successfully")

    #         return model_pusher_artifact

    #     except Exception as e:
    #         raise MyException(e, sys)
    
    
    def initiate_model_pusher(self):

        try:
            logging.info("Model pusher started")

            # Step 1: Load best model
            best_model = self.load_best_model()

            # Step 2: Save model and preprocessor locally
            self.push_model(best_model)

            # Step 3: Register best model in MLflow
            self.register_model()

            # Step 4: Return artifact
            model_pusher_artifact = ModelPusherArtifact(
                pushed_model_path=(
                    self.model_pusher_config.pushed_model_path
                ),
                pushed_preprocessor_path=(
                    self.model_pusher_config.pushed_preprocessor_path
                )
            )

            logging.info("Model pusher completed successfully")

            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys)