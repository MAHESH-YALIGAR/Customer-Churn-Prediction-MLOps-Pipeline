
import os
import sys
import json
from typing import Any

import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelTrainerConfig
from src.exception import MyException
from src.logger import logging

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Customer-Churn-Prediction2")
class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ) -> None:

        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def load_training_data(self) -> tuple[Any, Any]:

        try:
            logging.info("Loading transformed training data")

            X_train = joblib.load(
                self.data_transformation_artifact.train_features_path
            )

            y_train = joblib.load(
                self.data_transformation_artifact.train_target_path
            )

            if not hasattr(X_train, "shape") or len(X_train.shape) != 2:
                raise ValueError("Training features must be a 2D array")

            if not hasattr(y_train, "shape") or len(y_train.shape) != 1:
                raise ValueError("Training targets must be a 1D array")

            if X_train.shape[0] == 0:
                raise ValueError("Training data is empty")

            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError(
                    "Training features and targets have different row counts"
                )

            if len(set(y_train)) < 2:
                raise ValueError(
                    "Training targets must contain at least two classes"
                )

            logging.info(
                "Training data loaded. X_train: %s, y_train: %s",
                X_train.shape,
                y_train.shape,
            )

            return X_train, y_train

        except Exception as e:
            logging.exception("Error while loading training data")
            raise MyException(e, sys) from e

    def save_model(self, model: Any, model_path: str) -> None:

        try:
            model_dir = os.path.dirname(model_path)

            if model_dir:
                os.makedirs(model_dir, exist_ok=True)

            joblib.dump(model, model_path)

            logging.info("Model saved successfully: %s", model_path)

        except Exception as e:
            logging.exception("Error while saving model")
            raise MyException(e, sys) from e

    def initiate_model_training(self) -> ModelTrainerArtifact:

        try:
            logging.info("Model training started")

            X_train, y_train = self.load_training_data()

            # Set MLflow experiment
            mlflow.set_experiment("Customer-Churn-Prediction")

            models = (
                (
                    "logistic_regression",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                    self.model_trainer_config.logistic_regression_model_path,
                ),
                (
                    "decision_tree",
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        random_state=42,
                    ),
                    self.model_trainer_config.decision_tree_model_path,
                ),
                (
                    "random_forest",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                    self.model_trainer_config.random_forest_model_path,
                ),
            )

            # Store MLflow run IDs
            run_ids = {}

            for model_name, model, model_path in models:

                logging.info("Training %s", model_name)

                # Create an MLflow run for each model
                with mlflow.start_run(run_name=model_name) as run:

                    # Train model
                    model.fit(X_train, y_train)

                    # Capture MLflow run ID
                    run_ids[model_name] = run.info.run_id

                    # Log model parameters
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_params(model.get_params())

                    # Log trained model in MLflow
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        serialization_format="pickle",
                    )

                    # Save model locally
                    self.save_model(model, model_path)

                    logging.info(
                        "%s training completed. MLflow Run ID: %s",
                        model_name,
                        run_ids[model_name],
                    )

            run_ids_path = self.model_trainer_config.run_ids_path
            # logging.info("MLflow run IDs saved to %s", run_ids_path)
            run_ids_dir = os.path.dirname(run_ids_path)
            if run_ids_dir:
                os.makedirs(run_ids_dir, exist_ok=True)

            with open(run_ids_path, "w", encoding="utf-8") as file:
                json.dump(run_ids, file, indent=4)

            logging.info("MLflow run IDs saved to %s", run_ids_path)

            # Create ModelTrainerArtifact
            artifact = ModelTrainerArtifact(
                logistic_regression_model_path=(
                    self.model_trainer_config.logistic_regression_model_path
                ),
                decision_tree_model_path=(
                    self.model_trainer_config.decision_tree_model_path
                ),
                random_forest_model_path=(
                    self.model_trainer_config.random_forest_model_path
                ),
                logistic_regression_run_id=(
                    run_ids["logistic_regression"]
                ),
                decision_tree_run_id=(
                    run_ids["decision_tree"]
                ),
                random_forest_run_id=(
                    run_ids["random_forest"]
                ),
            )

            logging.info("All model training completed successfully")

            return artifact

        except Exception as e:
            logging.exception("Model training failed")
            raise MyException(e, sys) from e
