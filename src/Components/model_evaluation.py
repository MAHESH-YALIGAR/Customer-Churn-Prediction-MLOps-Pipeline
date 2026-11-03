
import os
import sys
import json
import joblib
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.logger import logging
from src.exception import MyException
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Customer-Churn-Prediction")
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)


class ModelEvaluation:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig,
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
            logging.exception("Error loading test data")
            raise MyException(e, sys) from e

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
                "confusion_matrix": cm,
            }

        except Exception as e:
            logging.exception("Error evaluating model")
            raise MyException(e, sys) from e

    def initiate_model_evaluation(self):

        try:
            logging.info("Model evaluation started")

            X_test, y_test = self.load_test_data()

            # Model paths and their corresponding MLflow run IDs
            models = {
                "Logistic Regression": {
                    "path": (
                        self.model_trainer_artifact
                        .logistic_regression_model_path
                    ),
                    "run_id": (
                        self.model_trainer_artifact
                        .logistic_regression_run_id
                    ),
                },

                "Decision Tree": {
                    "path": (
                        self.model_trainer_artifact
                        .decision_tree_model_path
                    ),
                    "run_id": (
                        self.model_trainer_artifact
                        .decision_tree_run_id
                    ),
                },

                "Random Forest": {
                    "path": (
                        self.model_trainer_artifact
                        .random_forest_model_path
                    ),
                    "run_id": (
                        self.model_trainer_artifact
                        .random_forest_run_id
                    ),
                },
            }

            mlflow.set_experiment("Customer-Churn-Prediction")

            evaluation_report = {}

            for model_name, model_info in models.items():

                logging.info("Evaluating %s", model_name)

                model = joblib.load(model_info["path"])

                metrics = self.evaluate_model(
                    model,
                    X_test,
                    y_test,
                )

                evaluation_report[model_name] = metrics
                logging.info(
    "Logging evaluation for run ID: %s",
    model_info["run_id"],
)
                # Reopen the original training run
                with mlflow.start_run(
                    run_id=model_info["run_id"]
                ):

                    # Log evaluation metrics into the same run
                    mlflow.log_metrics({
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                        and value is not None
                    })

                    # Log confusion matrix as an artifact
                    mlflow.log_dict(
                        {"confusion_matrix": metrics["confusion_matrix"]},
                        "evaluation/confusion_matrix.json",
                    )

                logging.info(
                    "%s F1 Score: %s",
                    model_name,
                    metrics["f1_score"],
                )

            # Select best model based on F1 score
            best_model_name = max(
                evaluation_report,
                key=lambda name: evaluation_report[name]["f1_score"],
            )

            best_model_score = evaluation_report[
                best_model_name
            ]["f1_score"]

            evaluation_report["best_model"] = {
                "name": best_model_name,
                "f1_score": best_model_score,
            }

            # Save evaluation report
            report_path = (
                self.model_evaluation_config.model_evaluation_report_path
            )

            os.makedirs(
                self.model_evaluation_config.model_evaluation_dir,
                exist_ok=True,
            )

            with open(report_path, "w") as file:
                json.dump(evaluation_report, file, indent=4)

            logging.info("Model evaluation completed successfully")
            logging.info("Best model: %s", best_model_name)

            return ModelEvaluationArtifact(
                model_evaluation_report_path=report_path,
                best_model_name=best_model_name,
                best_model_score=float(best_model_score),
            )

        except Exception as e:
            logging.exception("Model evaluation failed")
            raise MyException(e, sys) from e
