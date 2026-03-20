import pandas as pd
from src.inference.defaults import DEFAULT_CONTEXT, MODEL_FEATURES
from src.inference.lookup import get_gpu_specs


def fill_main_context(user_input):
    """
    Fill missing benchmark context fields with default values.

    Parameters
    ----------
    user_input : dict
        Dictionary of user supplied query values.

    Returns
    -------
    dict
        Copy of the input dictionary with missing Resolution and Setting
        fields filled using DEFAULT_CONTEXT.

    """
    completed = user_input.copy()

    if "Resolution" not in completed or completed["Resolution"] is None:
        completed["Resolution"] = DEFAULT_CONTEXT["Resolution"]

    if "Setting" not in completed or completed["Setting"] is None:
        completed["Setting"] = DEFAULT_CONTEXT["Setting"]

    return completed


def merge_inputs(specs_df, user_input):
    """
    Merge GPU specification values with user supplied overrides.

    Parameters
    ----------
    specs_dict : dict
        Dictionary of base GPU specification values, usually returned
        from GPU lookup.
    user_input : dict
        Dictionary of user supplied values that should override base
        specification values when keys overlap.

    Returns
    -------
    pandas.DataFrame
        One row dataframe containing merged values restricted to
        MODEL_FEATURES.
    """
    merged = specs_df.to_dict().copy()
    for key, value in user_input.items():
        merged[key] = value

    return pd.DataFrame([merged])[MODEL_FEATURES]


def prepare_features(user_input, df):
    """
    Build a complete feature dictionary for model inference.

    Parameters
    ----------
    user_input : dict
        Dictionary of user supplied query values. May include gpu_name,
        benchmark context fields, and manual feature overrides.
    df : pandas.DataFrame
        Dataframe containing GPU benchmark rows and specification data.

    Returns
    -------
    dict
        Dictionary containing one value for each feature listed in
        MODEL_FEATURES.

    Workflow
    --------
    1. Fill missing Resolution and Setting values.
    2. If gpu_name is provided, retrieve GPU specification values.
    3. Overlay user supplied values on top of looked up GPU values.
    4. Keep only keys present in MODEL_FEATURES.

    Notes
    -----
    If gpu_name is not provided, the returned feature dictionary is built
    only from user supplied values and default benchmark context fields.
    If a feature is still missing after merging, its value will be None.
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
