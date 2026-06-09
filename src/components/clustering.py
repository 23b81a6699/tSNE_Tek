import os
import sys
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object


class Clustering:

    def __init__(self):

        self.model_path = os.path.join(
            "artifacts",
            "kmeans.pkl"
        )

        self.cluster_path = os.path.join(
            "artifacts",
            "cluster_labels.csv"
        )

    def train_cluster_model(self, X):

        try:

            logger.info(
                "Training KMeans Model"
            )

            kmeans = KMeans(
                n_clusters=4,
                random_state=42,
                n_init=10
            )

            clusters = kmeans.fit_predict(X)

            score = silhouette_score(
                X,
                clusters
            )

            pd.DataFrame(
                {"Cluster": clusters}
            ).to_csv(
                self.cluster_path,
                index=False
            )

            save_object(
                self.model_path,
                kmeans
            )

            logger.info(
                "KMeans Training Completed"
            )

            return clusters, score

        except Exception as e:
            raise CustomException(
                e,
                sys
            )