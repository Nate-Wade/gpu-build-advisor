import pandas as pd
from src.data.load_data import load_final_data
from src.inference.lookup import get_observed_fps
from src.inference.features import prepare_features
from src.inference.reverse_lookup import find_gpus_from_fps
from src.models.predict import predict_fps_from_features


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
