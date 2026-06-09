import os
import sys

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object


class DataTransformation:

    def __init__(self):

        self.scaler_path = os.path.join(
            "artifacts",
            "scaler.pkl"
        )

        self.processed_path = os.path.join(
            "artifacts",
            "processed_data.csv"
        )

    def initiate_data_transformation(
        self,
        data_path
    ):

        try:

            logger.info(
                "Starting Data Transformation"
            )

            df = pd.read_csv(
                data_path
            )

            if "ID" in df.columns:
                df.drop(
                    columns=["ID"],
                    inplace=True
                )

            if "Dt_Customer" in df.columns:
                df.drop(
                    columns=["Dt_Customer"],
                    inplace=True
                )

            categorical_cols = df.select_dtypes(
                include=["object"]
            ).columns

            df = pd.get_dummies(
                df,
                columns=categorical_cols,
                drop_first=True
            )

            imputer = SimpleImputer(
                strategy="median"
            )

            X = imputer.fit_transform(df)

            scaler = StandardScaler()

            X_scaled = scaler.fit_transform(X)

            transformed_df = pd.DataFrame(
                X_scaled
            )

            transformed_df.to_csv(
                self.processed_path,
                index=False
            )

            save_object(
                self.scaler_path,
                scaler
            )

            logger.info(
                "Data Transformation Completed"
            )

            return X_scaled

        except Exception as e:
            raise CustomException(
                e,
                sys
            )