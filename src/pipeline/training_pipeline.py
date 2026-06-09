import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.append(ROOT_DIR)

from src.logger import logger

from src.components.data_ingestion import (
    DataIngestion
)

from src.components.data_transformation import (
    DataTransformation
)

from src.components.clustering import (
    Clustering
)

from src.components.tsne_model import (
    TSNEModel
)


class TrainingPipeline:

    def run_pipeline(self):

        logger.info(
            "Pipeline Started"
        )

        ingestion = DataIngestion()

        data_path = (
            ingestion.initiate_data_ingestion()
        )

        transformation = DataTransformation()

        X_scaled = (
            transformation
            .initiate_data_transformation(
                data_path
            )
        )

        clustering = Clustering()

        clusters, score = (
            clustering
            .train_cluster_model(
                X_scaled
            )
        )

        tsne_model = TSNEModel()

        embedding = (
            tsne_model
            .generate_embedding(
                X_scaled
            )
        )

        logger.info(
            f"Silhouette Score : {score}"
        )

        logger.info(
            f"Embedding Shape : {embedding.shape}"
        )

        print(
            "\nPipeline Completed Successfully"
        )

        print(
            f"Silhouette Score : {score:.4f}"
        )


if __name__ == "__main__":

    pipeline = TrainingPipeline()

    pipeline.run_pipeline()