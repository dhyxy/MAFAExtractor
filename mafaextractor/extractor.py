import numpy as np
import pandas as pd
from scipy.io import loadmat

from . import constants


def extract_mafa(
    matlab_file_path: str, dataset_type: str = None, clean: bool = True
) -> pd.DataFrame:
    """Extract the MAFA dataset's MATLAB file into a pandas DataFrame.

    Args:
        matlab_file_path (str): Path to MATLAB data file.
        dataset_type (str): "train" or "test", else it is inferred. Defaults to None.
        clean (bool, optional): Clean the dataframe to readable columns. Defaults to True.

    Raises:
        ValueError: If input dataset type is not "train" or "test".

    Returns:
        pd.DataFrame: Pandas DataFrame containing relevant information.
    """
    if not dataset_type:
        if "Train" in matlab_file_path or "train" in matlab_file_path:
            dataset_type = "train"
        elif "Test" in matlab_file_path or "test" in matlab_file_path:
            dataset_type = "test"
        else:
            dataset_type = ""
    dataset_type = dataset_type.lower()
    dataset_types = ("train", "test")
    if dataset_type not in dataset_types:
        raise ValueError(f"Dataset type must be in {dataset_types}")

    if dataset_type == "train":
        matlab_file_header = "label_train"
        img_names_header = "imgName"
        columns = constants.TRAIN_COLUMNS
    else:
        matlab_file_header = "LabelTest"
        img_names_header = "name"
        columns = constants.TEST_COLUMNS
    matlab_file = loadmat(matlab_file_path)
    data = matlab_file[matlab_file_header]
    img_names = data[img_names_header][0]
    labels = data["label"][0]
    data = make_data(img_names, labels)
    df = pd.DataFrame(data, columns=columns)
    if clean:
        df = clean_df_train(df) if dataset_type == "train" else clean_df_test(df)
    return df


def make_data(img_names: list, labels: list) -> list:
    """Format data appropriately for Pandas DataFrame.

    Args:
        img_names (list): Names of images.
        labels (list): Labels for images from dataset.

    Returns:
        list: List containing appropriate information for DataFrame.
    """
    rows = []
    for id_, img_name in enumerate(img_names):
        for label in labels[id_]:
            row = [img_name.item()]
            row.extend(label)
            rows.append(row)
    return rows


def clean_df_train(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame for training set.

    Args:
        df (pd.DataFrame): Raw training set DataFrame.

    Returns:
        pd.DataFrame: Clean training set DataFrame.
    """
    df.rename(
        columns={
            "x": "x_face_min",
            "y": "y_face_min",
            "x1": "left_eye_x",
            "y1": "left_eye_y",
            "x2": "right_eye_x",
            "y2": "right_eye_y",
            "w": "face_width",
            "h": "face_height",
            "w3": "occ_width",
            "h3": "occ_height",
            "w4": "glasses_width",
            "h4": "glasses_height",
        },
        inplace=True,
    )

    x_face_min = df["x_face_min"]
    y_face_min = df["y_face_min"]
    df["x_face_max"] = x_face_min + df["face_width"]
    df["y_face_max"] = y_face_min + df["face_height"]
    df["x_occ_min"] = x_face_min + df["x3"]
    df["y_occ_min"] = y_face_min + df["y3"]
    df["x_occ_max"] = x_face_min + df["occ_width"]
    df["y_occ_max"] = y_face_min + df["occ_height"]

    has_glasses = df["x4"] != -1
    df["x_glasses_min"] = np.where(has_glasses, x_face_min + df["x4"], -1)
    df["x_glasses_max"] = np.where(
        has_glasses, df["x_glasses_min"] + df["glasses_width"], -1
    )
    df["y_glasses_min"] = np.where(has_glasses, y_face_min + df["y4"], -1)
    df["y_glasses_max"] = np.where(
        has_glasses, df["y_glasses_min"] + df["glasses_height"], -1
    )

    df.drop(
        columns=["x3", "y3", "x4", "y4",], inplace=True,
    )
    return df


def clean_df_test(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame for test set.

    Args:
        df (pd.DataFrame): Raw test set DataFrame

    Returns:
        pd.DataFrame: Clean test set DataFrame.
    """
    df = df.rename(
        columns={
            "x": "x_face_min",
            "y": "y_face_min",
            "w": "face_width",
            "h": "face_height",
            "w1": "occ_width",
            "h1": "occ_height",
            "w2": "glasses_width",
            "h2": "glasses_height",
        }
    )
    x_face_min = df["x_face_min"]
    y_face_min = df["y_face_min"]
    df["x_face_max"] = x_face_min + df["face_width"]
    df["y_face_max"] = y_face_min + df["face_height"]
    df["x_occ_min"] = x_face_min + df["x1"]
    df["y_occ_min"] = y_face_min + df["y1"]
    df["x_occ_max"] = x_face_min + df["occ_width"]
    df["y_occ_max"] = y_face_min + df["occ_height"]

    has_glasses = df["x2"] != -1
    df["x_glasses_min"] = np.where(has_glasses, x_face_min + df["x2"], -1)
    df["x_glasses_max"] = np.where(
        has_glasses, df["x_glasses_min"] + df["glasses_width"], -1
    )
    df["y_glasses_min"] = np.where(has_glasses, y_face_min + df["y2"], -1)
    df["y_glasses_max"] = np.where(
        has_glasses, df["y_glasses_min"] + df["glasses_height"], -1
    )

    df.drop(
        columns=["x1", "y1", "x2", "y2"], inplace=True,
    )
    return df
