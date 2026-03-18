def validate_features(features, training_stats):

    warnings = []

    if features["Resolution"] not in training_stats["Resolution"]:
        warnings.append("Resolution not seen during training")

    if features["Setting"] not in training_stats["Setting"]:
        warnings.append("Setting not seen during training")

    if features["memory_size_GB"] > training_stats["memory_size_GB"]["max"]:
        warnings.append("Memory size outside training range")

    return warnings
