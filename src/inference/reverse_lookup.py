from src.inference.defaults import MODEL_FEATURES
from src.inference.features import fill_main_context
from src.models.predict import predict_fps_from_features


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
