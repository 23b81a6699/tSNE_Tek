import os
import sys

from sklearn.manifold import TSNE

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object


class TSNEModel:

    def __init__(self):

        self.embedding_path = os.path.join(
            "artifacts",
            "tsne_embedding.pkl"
        )

    def generate_embedding(
        self,
        X
    ):

        try:

            logger.info(
                "Generating t-SNE Embedding"
            )

            tsne = TSNE(
                n_components=2,
                perplexity=30,
                learning_rate="auto",
                init="pca",
                random_state=42
            )

            embedding = tsne.fit_transform(
                X
            )

            save_object(
                self.embedding_path,
                embedding
            )

            logger.info(
                "t-SNE Embedding Generated"
            )

            return embedding

        except Exception as e:
            raise CustomException(
                e,
                sys
            )