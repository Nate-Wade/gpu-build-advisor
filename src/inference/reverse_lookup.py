from src.inference.defaults import MODEL_FEATURES
from src.inference.features import fill_main_context
from src.models.predict import predict_fps_from_features
import pandas as pd


def find_gpus_from_fps(user_input, df, model):
    """
    Identify GPUs whose predicted FPS is closest to a target FPS value.

    Parameters
    ----------
    user_input : dict
        Dictionary of user supplied query values. Must include an "fps"
        key representing the target FPS. May also include benchmark
        context fields such as Game_Name, Resolution, and Setting.
    df : pandas.DataFrame
        Dataframe containing GPU benchmark rows and specification data.
        The dataframe index is expected to represent GPU names.
    model : object
        Trained model object implementing a predict method.

    Returns
    -------
    list of dict
        A list of dictionaries sorted by closeness to the target FPS.
        Each dictionary contains:
        - "gpu_name": canonical GPU name from the dataframe index
        - "predicted_fps": model predicted FPS for that GPU under the
          supplied benchmark context
        - "difference": absolute difference between predicted FPS and
          target FPS
    """

    target_fps = user_input["fps"]
    max_price = user_input.get("max_price")

    context_input = fill_main_context(user_input.copy())
    context_input.pop("fps", None)
    context_input.pop("max_price", None)

    results = []

    for gpu_name, gpu_rows in df.groupby(df.index):
        gpu_features = gpu_rows.iloc[0].to_dict()

        merged = gpu_features.copy()
        merged.update(context_input)

        features = {f: merged.get(f) for f in MODEL_FEATURES}
        pred = predict_fps_from_features(features, model)

        price = gpu_features.get("launch_price_USD")

        results.append({
            "gpu_name": gpu_name,
            "predicted_fps": pred,
            "price": price,
            "difference": abs(pred - target_fps)
        })

    meeting_target = [r for r in results if r["predicted_fps"] >= target_fps]

    if max_price is not None:
        meeting_target = [
            r for r in meeting_target
            if pd.notna(r["price"]) and r["price"] <= max_price
        ]

    if meeting_target:
        meeting_target.sort(key=lambda x: x["price"])
        return meeting_target[0]

    results.sort(key=lambda x: x["difference"])

    return results[0]
