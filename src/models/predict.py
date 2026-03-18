import pandas as pd
from src.inference.defaults import MODEL_FEATURES


def predict_fps_from_features(features, model):
    """
    Run model prediction given a complete feature dictionary.
    """
    df = pd.DataFrame([features], columns=MODEL_FEATURES)

    pred = model.predict(df)[0]

    return max(0.0, int(pred))
