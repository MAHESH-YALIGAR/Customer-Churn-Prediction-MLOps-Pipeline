import os


"""
This provides the source path for the data ingestion module.
"""
SOURCE_URL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "Notebook",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
)
RAW_DATA_DIR = "data/raw"
RAW_DATA_FILE_PATH = "data/raw/customer_churn.csv"


"""
Required columns for data validation.
"""
REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]
REPORT_FILE_PATH = "artifacts/data_validation_report.json"


"""
thi is for the data transfermation.py  file
"""

TRANSFORMED_DATA_DIR="artifacts/transformed"

TRAIN_FEATURES_PATH="artifacts/transformed/X_train.pkl"
TEST_FEATURES_PATH="artifacts/transformed/X_test.pkl"

TRAIN_TARGET_PATH="artifacts/transformed/y_train.pkl"
TEST_TARGET_PATH="artifacts/transformed/y_test.pkl"

PREPROCESSOR_FILE_PATH="artifacts/transformed/preprocessor.pkl"

TEST_SIZE= 0.2
RANDOM_STATE= 42


TRAINED_MODEL_DIR="artifacts/models"

LOGISTIC_REGRESSION_MODEL_PATH="artifacts/models/logistic_regression.pkl"

DECISION_TREE_MODEL_PATH="artifacts/models/decision_tree.pkl"

RANDOM_FOREST_MODEL_PATH="artifacts/models/random_forest.pkl"






MODEL_EVALUATION_DIR = "artifacts/model_evaluation"

MODEL_EVALUATION_REPORT_PATH = (
    "artifacts/model_evaluation/model_evaluation_report.json"
)



MODEL_PUSHER_DIR = "artifacts/model_pusher"

PUSHED_MODEL_PATH = "artifacts/model_pusher/model.pkl"

PUSHED_PREPROCESSOR_PATH = (
    "artifacts/model_pusher/preprocessor.pkl"
)