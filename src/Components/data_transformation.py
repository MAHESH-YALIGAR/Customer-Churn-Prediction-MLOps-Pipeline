import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import MyException
from src.logger import logging

from src.entity.artifact_entity import (DataIngestionArtifact,DataTransformationArtifact)
from src.entity.config_entity import DataTransformationConfig


class DataTransformation:
  def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_transformation_config:DataTransformationConfig):
    
    try:
      self.data_ingestion_artifact=data_ingestion_artifact
      self.data_transformation_config=data_transformation_config
    except Exception as e:
      raise MyException(e,sys)
    
    
  def read_data(self) -> pd.DataFrame:
        try:
            logging.info("Reading dataset for data transformation")

            file_path = self.data_ingestion_artifact.raw_data_file_path

            df = pd.read_csv(file_path)

            logging.info(f"Dataset loaded successfully: {df.shape}")

            return df

        except Exception as e:
            logging.exception("Error while reading dataset")
            raise MyException(e, sys)
  def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data transformation started")

            # Step 1: Read raw dataset
            df = self.read_data()

            # Step 2: Remove customerID because it is an identifier,
            # not a useful predictive feature.
            df = df.drop(columns=["customerID"])

            # Step 3: Clean TotalCharges
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True),
                errors="coerce"
            )

            # Step 4: Separate features and target
            X = df.drop(columns=["Churn"])

            y = df["Churn"].map({
                "No": 0,
                "Yes": 1
            })

            if y.isnull().any():
                raise ValueError("Unexpected values found in Churn column")

            # Step 5: Identify numerical and categorical columns
            numerical_columns = [
                "SeniorCitizen",
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            categorical_columns = [
                column for column in X.columns
                if column not in numerical_columns
            ]

            # Step 6: Create numerical preprocessing pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Step 7: Create categorical preprocessing pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            # Step 8: Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical", numerical_pipeline, numerical_columns),
                    ("categorical", categorical_pipeline, categorical_columns)
                ]
            )

            # Step 9: Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.data_transformation_config.test_size,
                random_state=self.data_transformation_config.random_state,
                stratify=y
            )

            # Step 10: Fit preprocessing only on training data
            X_train_transformed = preprocessor.fit_transform(X_train)

            # Use the same fitted preprocessing on test data
            X_test_transformed = preprocessor.transform(X_test)

            # Step 11: Create output directory
            os.makedirs(
                self.data_transformation_config.transformed_data_dir,
                exist_ok=True
            )

            # Step 12: Save transformed datasets
            train_features_path = (
                self.data_transformation_config.train_features_path
            )

            test_features_path = (
                self.data_transformation_config.test_features_path
            )

            train_target_path = (
                self.data_transformation_config.train_target_path
            )

            test_target_path = (
                self.data_transformation_config.test_target_path
            )

            preprocessor_path = (
                self.data_transformation_config.preprocessor_file_path
            )

            joblib.dump(X_train_transformed, train_features_path)
            joblib.dump(X_test_transformed, test_features_path)

            joblib.dump(y_train, train_target_path)
            joblib.dump(y_test, test_target_path)

            joblib.dump(preprocessor, preprocessor_path)

            logging.info("Transformed datasets saved successfully")

            # Step 13: Create transformation artifact
            transformation_artifact = DataTransformationArtifact(
                train_features_path=train_features_path,
                test_features_path=test_features_path,
                train_target_path=train_target_path,
                test_target_path=test_target_path,
                preprocessor_file_path=preprocessor_path
            )

            logging.info("Data transformation completed successfully")

            return transformation_artifact

        except Exception as e:
            logging.exception("Data transformation failed")
            raise MyException(e, sys)
       
          
      