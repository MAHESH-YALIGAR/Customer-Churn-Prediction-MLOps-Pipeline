from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
import os
import sys
import pandas as pd

class DataIngestion:
  def __init__(self,data_ingestion_config:DataIngestionConfig):
    try:
      self.data_ingestion_config=data_ingestion_config
    except Exception as e:
      raise MyException(e,sys)
  
  def get_data_from_source(self):    
    try:
      logging.info("Reading customer churn dataset from source")
      url=self.data_ingestion_config.source_url
      df = pd.read_csv(url)
      logging.info(
                f"Dataset successfully retrieved. Shape: {df.shape}"
            )
    except Exception as e:
      logging.exception("Error while saving raw dataset")
      raise MyException(e,sys)
    
  def save_data(self, df: pd.DataFrame) -> str:
        """
        Save the raw dataset into the configured directory.
        """
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            raw_data_file_path = self.data_ingestion_config.raw_data_file_path

            os.makedirs(raw_data_dir, exist_ok=True)

            logging.info("Saving raw dataset")

            df.to_csv(raw_data_file_path, index=False)

            logging.info(
                f"Raw dataset saved successfully at: {raw_data_file_path}"
            )

            return raw_data_file_path

        except Exception as e:
            logging.exception("Error while saving raw dataset")
            raise MyException(e, sys)


  def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Execute the complete data ingestion process.
        """
        try:
            logging.info("Data ingestion process started")

            # Step 1: Retrieve dataset
            df = self.get_data_from_source()

            # Step 2: Save raw dataset
            raw_data_file_path = self.save_data(df)

            # Step 3: Create ingestion artifact
            data_ingestion_artifact = DataIngestionArtifact(
                raw_data_file_path=raw_data_file_path
            )

            logging.info("Data ingestion completed successfully")

            return data_ingestion_artifact

        except Exception as e:
            logging.exception("Data ingestion process failed")
            raise MyException(e, sys)