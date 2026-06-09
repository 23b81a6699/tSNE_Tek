import os
import pandas as pd

from src.logger import logger
from src.exception import CustomException

import sys


class DataIngestion:

    def __init__(self):

        self.raw_data_path = os.path.join(
            "artifacts",
            "customer_data.csv"
        )

    def initiate_data_ingestion(self):

        try:

            logger.info(
                "Starting Data Ingestion"
            )

            os.makedirs(
                "artifacts",
                exist_ok=True
            )

            dataset_path = os.path.join(
                "data",
                "marketing_campaign.csv"
            )

            df = pd.read_csv(
                dataset_path,
                sep="\t"
            )

            df.to_csv(
                self.raw_data_path,
                index=False
            )

            logger.info(
                "Data Ingestion Completed"
            )

            return self.raw_data_path

        except Exception as e:
            raise CustomException(
                e,
                sys
            )