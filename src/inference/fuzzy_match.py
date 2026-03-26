import re
from difflib import get_close_matches
from src.inference.defaults import RESOLUTION_ALIASES, SETTING_ALIASES


def normalize_text(text):
    """
    Normalize text for matching by lowercasing, removing punctuation,
    and standardizing whitespace.
    """
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def fuzzy_match_value(user_value, valid_values, cutoff=0.6):
    """
    Return the closest match value from a list of valid values

    Parameters
    ------------
    user_value : str
        Raw user input
    valid_values : list
        List of valid values to match against
    cutoff : float
        Similarity threshold for accepting a match, between 0 and 1

    Returns
    --------
    str or None
        Best match if found, None otherwise
    """

    if user_value is None:
        return None

    normalized_map = {normalize_text(v): v for v in valid_values}
    normalized_choices = list(normalized_map.keys())

    matches = get_close_matches(
        normalize_text(user_value),
        normalized_choices,
        n=1,
        cutoff=cutoff
    )

    if not matches:
        return None

    return normalized_map[matches[0]]


def resolve_field(user_value, valid_values, cutoff=0.6, alias_map=None):
    """
    Resolve a user supplied value to a canonical value using:
    1. alias mapping
    2. exact normalized match
    3. fuzzy matching
    """
    if user_value is None:
        return None

    normalized_user = normalize_text(user_value)

    if alias_map and normalized_user in alias_map:
        return alias_map[normalized_user]

    for value in valid_values:
        if normalize_text(value) == normalized_user:
            return value

    return fuzzy_match_value(user_value, valid_values, cutoff=cutoff)


def resolve_gpu_name(user_value, valid_values, cutoff=0.5):
    """
    Resolve GPU names more carefully than plain fuzzy matching.

    Strategy:
    1. exact normalized match
    2. exact token match
    3. prefer shortest candidate containing all user tokens
    4. fuzzy fallback
    """
    if user_value is None:
        return None

    normalized_user = normalize_text(user_value)
    user_tokens = normalized_user.split()

    normalized_pairs = [(normalize_text(v), v) for v in valid_values]

    # 1. exact normalized match
    for norm, original in normalized_pairs:
        if norm == normalized_user:
            return original

    # 2. exact token match
    token_matches = []
    for norm, original in normalized_pairs:
        norm_tokens = norm.split()

        if all(token in norm_tokens for token in user_tokens):
            token_matches.append((norm, original))

    if token_matches:
        # Prefer the shortest normalized match
        # e.g. "rtx 3080" over "rtx 3080 ti"
        token_matches.sort(key=lambda x: len(x[0].split()))
        return token_matches[0][1]

    # 3. fuzzy fallback
    return fuzzy_match_value(user_value, valid_values, cutoff=cutoff)


def resolve_user_input(user_input, df):
    """
    Resolve user input fields into canonical dataset values.
    """
    resolved = user_input.copy()

    if resolved.get("gpu_name"):
        valid_gpus = df.index.astype(str).unique().tolist()
        matched_gpu = resolve_gpu_name(
            resolved["gpu_name"], valid_gpus, cutoff=0.5)
        if matched_gpu:
            resolved["gpu_name"] = matched_gpu
            resolved["max_price"] = int(
                df.loc[matched_gpu, "launch_price_USD"].iloc[0])

    if resolved.get("Game_Name"):
        valid_games = df["Game_Name"].dropna().astype(str).unique().tolist()
        matched_game = resolve_field(
            resolved["Game_Name"], valid_games, cutoff=0.5)
        if matched_game:
            resolved["Game_Name"] = matched_game

    if resolved.get("Resolution"):
        matched_resolution = resolve_field(
            resolved["Resolution"],
            ["1080p", "1440p", "4K"],
            cutoff=0.4,
            alias_map=RESOLUTION_ALIASES
        )
        if matched_resolution:
            resolved["Resolution"] = matched_resolution

    if resolved.get("Setting"):
        matched_setting = resolve_field(
            resolved["Setting"],
            ["Low", "Medium", "High", "Ultra"],
            cutoff=0.4,
            alias_map=SETTING_ALIASES
        )
        if matched_setting:
            resolved["Setting"] = matched_setting

    return resolved
