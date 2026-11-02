import os
import sys
import json
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.logger import logging
from src.exception import MyException

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)


class ModelEvaluation:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config

    def load_test_data(self):

        try:
            logging.info("Loading test data")

            X_test = joblib.load(
                self.data_transformation_artifact.test_features_path
            )

            y_test = joblib.load(
                self.data_transformation_artifact.test_target_path
            )

            logging.info("Test data loaded successfully")

            return X_test, y_test

        except Exception as e:
            raise MyException(e, sys)

    def evaluate_model(self, model, X_test, y_test):

        try:
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test, y_pred, zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, zero_division=0
            )
            f1 = f1_score(
                y_test, y_pred, zero_division=0
            )

            # Calculate ROC-AUC using prediction probabilities
            if hasattr(model, "predict_proba"):
                y_probability = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_probability)
            else:
                roc_auc = None

            cm = confusion_matrix(y_test, y_pred).tolist()

            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": (
                    float(roc_auc) if roc_auc is not None else None
                ),
                "confusion_matrix": cm
            }

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self):

        try:
            logging.info("Model evaluation started")

            # Load test data
            X_test, y_test = self.load_test_data()

            # Get trained model paths
            models = {
                "Logistic Regression":
                    self.model_trainer_artifact.logistic_regression_model_path,

                "Decision Tree":
                    self.model_trainer_artifact.decision_tree_model_path,

                "Random Forest":
                    self.model_trainer_artifact.random_forest_model_path
            }

            evaluation_report = {}

            for model_name, model_path in models.items():

                logging.info(f"Evaluating {model_name}")

                # Load trained model
                model = joblib.load(model_path)

                # Evaluate model
                metrics = self.evaluate_model(
                    model,
                    X_test,
                    y_test
                )

                evaluation_report[model_name] = metrics

                logging.info(
                    f"{model_name} F1 Score: {metrics['f1_score']}"
                )

            # Select best model based on F1 score
            best_model_name = max(
                evaluation_report,
                key=lambda name: evaluation_report[name]["f1_score"]
            )

            best_model_score = evaluation_report[
                best_model_name
            ]["f1_score"]

            evaluation_report["best_model"] = {
                "name": best_model_name,
                "f1_score": best_model_score
            }

            # Save evaluation report
            report_path = (
                self.model_evaluation_config.model_evaluation_report_path
            )

            os.makedirs(
                self.model_evaluation_config.model_evaluation_dir,
                exist_ok=True
            )

            with open(report_path, "w") as file:
                json.dump(evaluation_report, file, indent=4)

            logging.info("Model evaluation completed successfully")
            logging.info(f"Best model: {best_model_name}")

            return ModelEvaluationArtifact(
                model_evaluation_report_path=report_path,
                best_model_name=best_model_name,
                best_model_score=float(best_model_score)
            )

        except Exception as e:
            raise MyException(e, sys)