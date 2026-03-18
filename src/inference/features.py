import pandas as pd
from src.inference.defaults import DEFAULT_CONTEXT, MODEL_FEATURES
from src.inference.lookup import get_gpu_specs


def fill_main_context(user_input):
    """
    Ensure Resolution and Setting exist.
    """
    completed = user_input.copy()

    if "Resolution" not in completed or completed["Resolution"] is None:
        completed["Resolution"] = DEFAULT_CONTEXT["Resolution"]

    if "Setting" not in completed or completed["Setting"] is None:
        completed["Setting"] = DEFAULT_CONTEXT["Setting"]

    return completed


def merge_inputs(specs_df, user_input):
    merged = specs_df.to_dict().copy()
    print(merged)
    print(user_input)
    for key, value in user_input.items():
        merged[key] = value

    return pd.DataFrame([merged])[MODEL_FEATURES]


def prepare_features(user_input, df):
    """
    Build model features from user input and GPU specs.
    """

    user_input = fill_main_context(user_input)

    merged = {}

    # If GPU provided, pull specs
    if "gpu_name" in user_input:
        gpu_specs = get_gpu_specs(user_input["gpu_name"], df)
        merged.update(gpu_specs)

    # User values override lookup
    merged.update(user_input)

    # Keep only model features
    features = {}

    for feature in MODEL_FEATURES:
        features[feature] = merged.get(feature)

    return features
