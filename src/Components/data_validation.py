import os
import sys
import json
import pandas as pd

from src.logger import logging
from src.exception import MyException
from src.entity.artifact_entity import (DataIngestionArtifact,DataValidationArtifact)
from src.entity.config_entity import DataIngestionConfig


class DataValidation:
  def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationArtifact):
    
    try:
      self.data_ingestion_artifact=data_ingestion_artifact
      self.data_validation_config=data_validation_config
    except Exception as e:
      raise MyException(e,sys)
    
  def read_data(self) -> pd.DataFrame:
        """
        Read the raw dataset using the path from DataIngestionArtifact.
        """
        try:
            logging.info("Reading raw dataset for data validation")

            raw_data_path = (
                self.data_ingestion_artifact.raw_data_file_path
            )

            df = pd.read_csv(raw_data_path)

            logging.info(
                f"Dataset loaded successfully. Shape: {df.shape}"
            )

            return df

        except Exception as e:
            logging.exception("Failed to read raw dataset")
            raise MyException(e, sys)

  def validate_columns(self, df: pd.DataFrame) -> bool:
        """
        Check whether all required columns are present.
        """
        try:
            logging.info("Checking required columns")

            required_columns = (
                self.data_validation_config.required_columns
            )

            missing_columns = [
                col for col in required_columns
                if col not in df.columns
            ]

            if missing_columns:
                logging.error(
                    f"Missing required columns: {missing_columns}"
                )
                return False

            logging.info("All required columns are present")

            return True

        except Exception as e:
            logging.exception("Column validation failed")
            raise MyException(e, sys)

  def validate_data_types(self, df: pd.DataFrame) -> bool:
    try:
        logging.info("Checking data types")

        numeric_columns = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        invalid_columns = []

        for column in numeric_columns:

            if column not in df.columns:
                invalid_columns.append(column)
                continue

            if column == "TotalCharges":
                converted = pd.to_numeric(
                    df[column].replace(r"^\s*$", pd.NA, regex=True),
                    errors="coerce"
                )

                invalid_values = (
                    df[column].notna() & converted.isna()
                )

                if invalid_values.any():
                    invalid_columns.append(column)

            elif not pd.api.types.is_numeric_dtype(df[column]):
                invalid_columns.append(column)

        if invalid_columns:
            logging.error(
                f"Invalid numeric columns: {invalid_columns}"
            )
            return False

        logging.info("Data type validation successful")
        return True

    except Exception as e:
        logging.exception("Data type validation failed")
        raise MyException(e, sys)

  def validate_missing_values(self, df: pd.DataFrame) -> bool:
    try:
        logging.info("Checking missing values")

        missing_values = df.isnull().sum()

        # Also detect empty and whitespace-only strings
        blank_values = (
            df.select_dtypes(include=["object"])
            .apply(lambda col: col.str.strip().eq("").sum())
        )

        missing_columns = {}

        for column in df.columns:
            count = int(missing_values[column])

            if column in blank_values.index:
                count += int(blank_values[column])

            if count > 0:
                missing_columns[column] = count

        if missing_columns:
            logging.warning(
                f"Missing or blank values found: {missing_columns}"
            )
            return False

        logging.info("No missing values found")
        return True

    except Exception as e:
        logging.exception("Missing value validation failed")
        raise MyException(e, sys)
  def validate_duplicates(self, df: pd.DataFrame) -> bool:
        """
        Check whether duplicate records exist.
        """
        try:
            logging.info("Checking duplicate records")

            duplicate_count = int(df.duplicated().sum())

            if duplicate_count > 0:
                logging.warning(
                    f"Found {duplicate_count} duplicate records"
                )
                return False

            logging.info("No duplicate records found")

            return True

        except Exception as e:
            logging.exception("Duplicate validation failed")
            raise MyException(e, sys)

  def save_validation_report(self, report: dict) -> str:
        """
        Save validation results into a JSON file.
        """
        try:
            report_path = (
                self.data_validation_config.report_file_path
            )

            report_dir = os.path.dirname(report_path)

            if report_dir:
                os.makedirs(report_dir, exist_ok=True)

            with open(report_path, "w") as file:
                json.dump(report, file, indent=4)

            logging.info(
                f"Validation report saved at: {report_path}"
            )

            return report_path

        except Exception as e:
            logging.exception("Failed to save validation report")
            raise MyException(e, sys)

  def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Execute the complete data validation process.
        """
        try:
            logging.info("Data validation process started")

            # Step 1: Read dataset
            df = self.read_data()

            # Step 2: Validate dataset
            column_status = self.validate_columns(df)

            data_type_status = self.validate_data_types(df)

            missing_value_status = self.validate_missing_values(df)

            duplicate_status = self.validate_duplicates(df)

            # Step 3: Overall validation status
            validation_status = all([
                column_status,
                data_type_status,
                missing_value_status,
                duplicate_status
            ])

            # Step 4: Create validation report
            report = {
                "dataset_shape": {
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1])
                },
                "column_validation": column_status,
                "data_type_validation": data_type_status,
                "missing_value_validation": missing_value_status,
                "duplicate_validation": duplicate_status,
                "validation_status": validation_status
            }

            # Step 5: Save report
            report_path = self.save_validation_report(report)

            logging.info(
                f"Data validation completed. Status: {validation_status}"
            )

            # Step 6: Create artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                validation_report_file_path=report_path
            )

            return data_validation_artifact

        except Exception as e:
            logging.exception("Data validation process failed")
            raise MyException(e, sys)