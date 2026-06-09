import os
import pickle

from src.logger import logger


def save_object(file_path, obj):
    """
    Save any Python object as pickle.
    """

    try:
        dir_path = os.path.dirname(file_path)

        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logger.info(
            f"Object saved successfully at {file_path}"
        )

    except Exception as e:
        logger.error(
            f"Error while saving object: {str(e)}"
        )
        raise


def load_object(file_path):
    """
    Load pickle object.
    """

    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logger.info(
            f"Object loaded successfully from {file_path}"
        )

        return obj

    except Exception as e:
        logger.error(
            f"Error while loading object: {str(e)}"
        )
        raise


def create_directory(path):
    """
    Create directory if not exists.
    """

    os.makedirs(path, exist_ok=True)

    logger.info(
        f"Directory created: {path}"
    )


def save_dataframe(df, file_path):
    """
    Save DataFrame as CSV.
    """

    try:
        dir_path = os.path.dirname(file_path)

        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        df.to_csv(
            file_path,
            index=False
        )

        logger.info(
            f"DataFrame saved at {file_path}"
        )

    except Exception as e:
        logger.error(
            f"Error saving dataframe: {str(e)}"
        )
        raise