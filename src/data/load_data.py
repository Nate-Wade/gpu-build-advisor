from pathlib import Path
import pandas as pd
import joblib


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"


def load_gpu_spec_data():
    """
    Load the GPU specification dataset.

    Returns
    -------
    dict
        Dictionary of GPU specs indexed by GPU name.
    """
    path = DATA_DIR / "raw" / "gpu_specs_original.csv"
    df = pd.read_csv(path, index_col=0)
    return df.to_dict(orient="index")


def load_fps_data():
    """
    Load the benchmark FPS dataset.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing benchmark FPS measurements.
    """
    path = DATA_DIR / "raw" / "gpu_fps_only.csv"
    return pd.read_csv(path, index_col=0)


def load_final_data():
    """
    Load the final processed dataset used for training.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset used by the ML model.
    """
    path = DATA_DIR / "processed" / "gpu_data_final.csv"
    return pd.read_csv(path, index_col=0)


def load_model():
    """
    Load the trained XGBoost model.

    Returns
    -------
    object
        Trained model object.
    """
    path = MODEL_DIR / "xgb_gpu_fps_model.joblib"
    return joblib.load(path)
