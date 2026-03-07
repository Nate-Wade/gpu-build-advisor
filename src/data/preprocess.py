import pandas as pd
import numpy as np

from src.data.load_data import load_gpu_spec_data, load_fps_data

# EASY CLEANING PAIRS
COL_UNIT_PAIRS = {'process_size': 'nm',
                  'transistors': 'million',
                  'die_size': 'mm²',
                  'base_clock': 'MHz',
                  'memory_bus': 'bit',
                  'tdp': 'W',
                  'memory_clock': 'MHz',
                  'pixel_rate': 'GPixel/s',
                  'texture_rate': 'GTexel/s',
                  'launch_price': 'USD'}

# Gather Data


def load_data():
    """
    Load the GPU specifications data from a CSV file, 
    set the column names, and return it as a DataFrame.
    """

    gpu_specs = load_gpu_spec_data()
    gpu_specs_df = pd.DataFrame.from_dict(gpu_specs, orient='index')

    # Select Columns to keep
    gpu_specs_df = gpu_specs_df[['architecture', 'process_size', 'transistors', 'density', 'die_size',
                                'base_clock', 'memory_size', 'memory_type', 'memory_bus',
                                 'bandwidth', 'shading_units', 'tmus', 'rops', 'l1_cache',
                                 'l2_cache', 'directx', 'gpu_clock', 'tdp', 'memory_clock',
                                 'fp32_(float)', 'fp64_(double)', 'pixel_rate',
                                 'texture_rate', 'launch_price']]

    # Set name to index
    gpu_specs_df.index.name = 'name'

    # Remove Row due to outlier data
    gpu_specs_df = gpu_specs_df.drop(index="AMD Radeon RX VEGA 10")
    gpu_specs_df = gpu_specs_df.dropna(subset=['launch_price'])
    return gpu_specs_df


def clean_numeric_column(df, col, unit):
    # Replace "Unknown", "N/A", "-", None, etc. with NaN
    df[col] = df[col].replace(["Unknown", "N/A", "-", "", 'unknown'], pd.NA)

    # Remove commas and unit suffix
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(f" {unit}", "", regex=False)
    )

    # Convert to numeric safely (anything invalid -> NaN)
    df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rename column to include the unit
    df.rename(columns={col: f"{col}_{unit}"}, inplace=True)

    return df


def normalize_flops(df, column, new_name):
    # 1. Clean commas
    df[column] = df[column].astype(str).str.replace(",", "")

    # 2. Extract numeric value
    df[f"{column}_value"] = df[column].str.extract(r"([\d\.]+)").astype(float)

    # 3. Extract unit (GFLOPS, TFLOPS)
    df[f"{column}_unit"] = df[column].str.extract(r"([A-Za-z]+)")

    # 4. Convert GFLOPS → TFLOPS
    df.loc[df[f"{column}_unit"].str.upper().str.startswith("G"),
           f"{column}_value"] /= 1000

    # 5. Round to 2 decimals
    df[f"{column}_value"] = df[f"{column}_value"].round(2)

    # 6. Insert normalized column
    col_idx = df.columns.get_loc(column)
    df.insert(col_idx, new_name, df[f"{column}_value"])

    # 7. Drop helper columns
    df.drop(columns=[f"{column}_value",
            f"{column}_unit", column], inplace=True)

    return df


def clean_directx(val):
    if "Ultimate" in str(val):
        return "12.1"
    return val


def clean_numeric_columns(gpu_specs_df):
    # Easy numeric cleaning
    gpu_specs_df['base_clock'] = gpu_specs_df['base_clock'].fillna(
        gpu_specs_df['gpu_clock'])
    gpu_specs_df = gpu_specs_df.drop(columns=['gpu_clock'])
    for key, value in COL_UNIT_PAIRS.items():
        gpu_specs_df = clean_numeric_column(gpu_specs_df, key, value)
    gpu_specs_df = normalize_flops(gpu_specs_df, "fp32_(float)", "fp32_TFLOPS")
    gpu_specs_df = normalize_flops(
        gpu_specs_df, "fp64_(double)", "fp64_TFLOPS")

    return gpu_specs_df


def other_cleaning(gpu_specs_df):
    # More complicated numeric cleaning

    # Density
    unit = 'M / mm²'
    col = 'density'
    gpu_specs_df['density'] = (
        gpu_specs_df['density']
        .str.replace(",", "", regex=False)
        .str.replace(f"{unit}", "", regex=False)
        .astype(float)
    )
    # Add unit to the column name
    gpu_specs_df.rename(columns={
                        col: f"{col}_{unit.replace(' ', '_').replace('/', '_per_').replace('²', '^2')}"}, inplace=True)

    # Memory Size
    gpu_specs_df["memory_value"] = gpu_specs_df["memory_size"].str.extract(
        r"([\d\.]+)").astype(float)
    gpu_specs_df["memory_unit"] = gpu_specs_df["memory_size"].str.extract(
        r"([A-Za-z]+)")
    col_idx = gpu_specs_df.columns.get_loc("memory_size")
    gpu_specs_df.loc[gpu_specs_df["memory_unit"]
                     == "MB", "memory_value"] /= 1024
    gpu_specs_df.insert(col_idx, "memory_size_GB",
                        gpu_specs_df["memory_value"])
    gpu_specs_df = gpu_specs_df.drop(
        columns=["memory_value", "memory_unit", "memory_size"])

    # Bandwidth
    # Extract the unit (GB/s or TB/s)
    gpu_specs_df["bandwidth_value"] = gpu_specs_df["bandwidth"].str.extract(
        r"([\d\.]+)").astype(float)
    gpu_specs_df["bandwidth_unit"] = gpu_specs_df["bandwidth"].str.extract(
        r"([A-Za-z/]+)")
    # Find the column index of 'bandwidth'
    col_idx = gpu_specs_df.columns.get_loc("bandwidth")
    # Convert TB/s → GB/s
    gpu_specs_df.loc[gpu_specs_df["bandwidth_unit"]
                     == "TB/s", "bandwidth_value"] *= 1024
    # Insert normalized column (all values in GB/s)
    gpu_specs_df.insert(col_idx, "bandwidth_GBs",
                        gpu_specs_df["bandwidth_value"])
    # Drop intermediate columns
    gpu_specs_df = gpu_specs_df.drop(
        columns=["bandwidth_value", "bandwidth_unit", "bandwidth"])

    # l1 Cache
    col = 'l1_cache'
    gpu_specs_df[col] = gpu_specs_df[col].str.extract(
        r"([\d\.]+)").astype(float)
    # Rename with unit + qualifier
    gpu_specs_df.rename(columns={col: f"{col}_KB_per_CU"}, inplace=True)

    # l2 Cache
    gpu_specs_df["cache_value"] = gpu_specs_df["l2_cache"].str.extract(
        r"([\d\.]+)").astype(float)
    gpu_specs_df["cache_unit"] = gpu_specs_df["l2_cache"].str.extract(
        r"([A-Za-z]+)")
    col_idx = gpu_specs_df.columns.get_loc("l2_cache")
    gpu_specs_df.loc[gpu_specs_df["cache_unit"] == "KB", "cache_value"] /= 1024
    gpu_specs_df.insert(col_idx, "l2_cache_MB", gpu_specs_df["cache_value"])
    gpu_specs_df = gpu_specs_df.drop(
        columns=["cache_value", "cache_unit", "l2_cache"])

    # DirectX
    gpu_specs_df['directx'] = gpu_specs_df['directx'].apply(
        clean_directx).astype(float)

    # Fill Lingering NaN
    gpu_specs_df["l1_cache_KB_per_CU"] = gpu_specs_df["l1_cache_KB_per_CU"].fillna(
        0)
    gpu_specs_df["fp64_TFLOPS"] = gpu_specs_df["fp64_TFLOPS"].fillna(0)
    gpu_specs_df["tdp_W"] = gpu_specs_df["tdp_W"].fillna(0)

    return gpu_specs_df


def group_top_categories(df, column, top_n=5, new_col_name=None):
    """
    Group categorical column into top_n categories + 'Other'.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    column : str
        Column name to group.
    top_n : int
        Number of most frequent categories to keep.
    new_col_name : str or None
        Name for new column. If None, overwrites the original column.

    Returns
    -------
    DataFrame : updated DataFrame with grouped column.
    """
    if new_col_name is None:
        new_col_name = column

    # Get top N categories
    top_categories = df[column].value_counts().nlargest(top_n).index

    # Map values
    df[new_col_name] = df[column].apply(
        lambda x: x if x in top_categories else "Other")

    return df


def combine_fps_specs(df):
    fps_df = load_fps_data()
    fps_df['Avg_FPS'] = fps_df['Avg_FPS'].str.replace(
        ",", "", regex=False).astype(float)
    fps_df = fps_df.drop(columns="Min_FPS")
    gpu_joined = df.join(fps_df, how='inner')
    gpu_joined[gpu_joined.select_dtypes(include=['number']).columns] = gpu_joined.select_dtypes(
        include=['number']).apply(lambda x: x.astype(float))
    gpu_joined.to_csv("data/processed/gpu_data_final.csv", index=True)

    return gpu_joined


def preprocess_gpu_data():
    gpu_specs_df = load_data()
    gpu_specs_df = clean_numeric_columns(gpu_specs_df)
    gpu_specs_df = other_cleaning(gpu_specs_df)
    gpu_specs_df = group_top_categories(gpu_specs_df, "architecture", top_n=5)
    gpu_specs_df = group_top_categories(gpu_specs_df, "memory_type", top_n=5)

    gpu_final = combine_fps_specs(gpu_specs_df)
    return gpu_final
