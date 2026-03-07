import pandas as pd


def load_gpu_spec_data():
    return pd.read_csv("data/raw/gpu_specs_original.csv", index_col=0).to_dict(orient="index")


def load_fps_data():
    return pd.read_csv("data/raw/gpu_fps_only.csv", index_col=0)
