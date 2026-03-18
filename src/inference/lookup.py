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
