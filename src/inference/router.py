import pandas as pd
from src.data.load_data import load_final_data
from src.inference.lookup import get_observed_fps
from src.inference.features import prepare_features
from src.inference.reverse_lookup import find_gpus_from_fps
from src.models.predict import predict_fps_from_features
from src.inference.fuzzy_match import resolve_user_input


def answer_fps_query(user_input, model):
    """
    Resolve a user FPS query using observed data, model prediction,
    or reverse lookup depending on the input.

    Parameters
    ----------
    user_input : dict
        Dictionary of user supplied query values. May include:
        - "gpu_name" for forward prediction
        - benchmark context fields such as Game_Name, Resolution, Setting
        - "fps" for reverse lookup mode
    model : object
        Trained model object implementing a predict method.

    Returns
    -------
    dict
        Dictionary containing the result and source of the response.

        Forward prediction or observed lookup:
        {
            "fps": int,
            "source": "observed" or "predicted"
        }

        Reverse lookup:
        {
            "matches": list of dict,
            "source": "reverse_lookup"
        }
    """

    df = load_final_data()

    # Fuzzy Matching
    user_input = resolve_user_input(user_input, df)

    # Reverse lookup mode
    if "fps" in user_input:
        return find_gpus_from_fps(user_input, df, model)

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
