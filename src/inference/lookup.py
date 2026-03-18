def get_gpu_specs(gpu_name, df):
    """
    Return GPU specification values for a given GPU name.

    Parameters
    ----------
    gpu_name : str
        Exact GPU name to look up in the dataframe index.
    df : pandas.DataFrame
        Dataframe containing GPU benchmark rows indexed by GPU name.

    Returns
    -------
    dict
        Dictionary of GPU specification values from the first matching row.

    Raises
    ------
    ValueError
        If the GPU name is not found in the dataframe index.
    """
    try:
        return df.loc[gpu_name].iloc[0].to_dict()
    except KeyError:
        raise ValueError(f"GPU not found: {gpu_name}")


def get_observed_fps(user_input, df):
    """
    Return an observed FPS value from the dataset using best row match scoring.

    Parameters
    ----------
    user_input : dict
        Dictionary of user supplied input values such as gpu_name, Game_Name,
        Resolution, Setting, or other model related fields.
    df : pandas.DataFrame
        Dataframe containing benchmark rows and an Avg_FPS column.

    Returns
    -------
    int or None
        Observed FPS from the best matching row if a meaningful match is found.
        Returns None if no row matches any provided fields.

    """

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
