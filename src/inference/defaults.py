"""
File for defining constants and default values used across the inference process.
"""

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


RESOLUTION_ALIASES = {
    "1080": "1920x1080",
    "1080p": "1920x1080",
    "1920x1080": "1920x1080",

    "1440": "2560x1440",
    "1440p": "2560x1440",
    "2k": "2560x1440",
    "2560x1440": "2560x1440",

    "4k": "3840x2160",
    "2160": "3840x2160",
    "2160p": "3840x2160",
    "uhd": "3840x2160",
    "3840x2160": "3840x2160"
}

SETTING_ALIASES = {
    "low": "Low",
    "low settings": "Low",
    "medium": "Medium",
    "medium settings": "Medium",
    "high": "High",
    "high settings": "High",
    "ultra": "Ultra",
    "ultra settings": "Ultra",
    "max": "Ultra",
    "max settings": "Ultra"
}
