import os
import sys
from typing import Any

import joblib
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
                raise ValueError("Training targets must contain at least two classes")

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

            for model_name, model, model_path in models:
                logging.info("Training %s", model_name)
                model.fit(X_train, y_train)
                self.save_model(model, model_path)

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
            )
            logging.info("All model training completed successfully")
            return artifact

        except Exception as e:
            logging.exception("Model training failed")
            raise MyException(e, sys) from e
