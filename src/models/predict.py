import pandas as pd
from src.data.load_data import load_final_data

MODEL_FEATURES = [
    "architecture",
    "process_size_nm",
    "transistors_million",
    "density_M__per__mm^2",
    "die_size_mm²",
    "base_clock_MHz",
    "memory_size_GB",
    "memory_type",
    "memory_bus_bit",
    "bandwidth_GBs",
    "shading_units",
    "tmus",
    "rops",
    "l1_cache_KB_per_CU",
    "l2_cache_MB",
    "directx",
    "tdp_W",
    "memory_clock_MHz",
    "fp32_TFLOPS",
    "fp64_TFLOPS",
    "pixel_rate_GPixel/s",
    "texture_rate_GTexel/s",
    "Game_Name",
    "Avg_FPS",
    "Setting",
    "Resolution",
    "launch_price_USD"
]

DEFAULT_CONTEXT = {
    "Resolution": "1920x1080",
    "Setting": "Medium"
}


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


def find_gpus_from_fps(user_input, df, model, top_n=5):

    target_fps = user_input["fps"]

    user_input = fill_main_context(user_input)
    user_input.pop("fps", None)

    results = []

    for gpu_name, gpu_row in df.groupby(df.index):

        gpu_features = gpu_row.iloc[0].to_dict()

        merged = gpu_features.copy()
        merged.update(user_input)

        features = {f: merged.get(f) for f in MODEL_FEATURES}

        pred = predict_fps_from_features(features, model)

        results.append({
            "gpu_name": gpu_name,
            "predicted_fps": pred,
            "difference": abs(pred - target_fps)
        })

    results.sort(key=lambda x: x["difference"])

    return results[:top_n]


def get_gpu_specs(gpu_name, df):
    """
    Retrieve GPU specifications by name.
    """
    try:
        return df.loc[gpu_name].iloc[0].to_dict()
    except KeyError:
        raise ValueError(f"GPU not found: {gpu_name}")


def get_observed_fps(user_input, df):

    rows = df.copy()
    rows["match_score"] = 0

    for key, value in user_input.items():
        if key == "gpu_name":
            rows["match_score"] += (
                rows.index.astype(str).str.lower() == str(value).lower()
            ).astype(int)
        elif key in rows.columns:
            rows["match_score"] += (
                rows[key].astype(str).str.lower() == str(value).lower()
            ).astype(int)

    row = rows.sort_values("match_score", ascending=False).iloc[0]

    print(row)
    return int(row["Avg_FPS"])


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


def predict_fps_from_features(features, model):
    df = pd.DataFrame([features], columns=MODEL_FEATURES)
    pred = model.predict(df)[0]
    return max(0.0, int(pred))


def answer_fps_query(user_input, model):

    df = load_final_data()

    # Reverse lookup mode
    if "fps" in user_input:
        return {
            "matches": find_gpus_from_fps(user_input, df, model),
            "source": "reverse_lookup"
        }

    # Try observed benchmark first
    observed = get_observed_fps(user_input, df)

    if observed is not None:
        return {
            "fps": observed,
            "source": "observed"
        }

    # Predict
    features = prepare_features(user_input, df)

    predicted = predict_fps_from_features(features, model)

    return {
        "fps": predicted,
        "source": "predicted"
    }
