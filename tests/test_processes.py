from src.data.preprocess import preprocess_gpu_data
from src.models.train_model import train
from src.data.load_data import load_model
from src.inference.router import answer_fps_query

model = load_model()
user_input = {
    "gpu_name": "NVIDIA GeForce RTX 3080",
    "Game_Name": "Fortnite Battle Royale",
    "Resolution": "2560x1440",
    "Setting": "Medium"

}

user_input_game_gpu = {
    "gpu_name": "NVIDIA GeForce RTX 3080",
    "Game_Name": "Fortnite Battle Royale",

}

user_input_spec = {
    "architecture": "Ampere",
    "memory_size_GB": 10,
    "Game_Name": "Cyberpunk 2077"
}

user_input_fps = {
    "Game_Name": "Fortnite Battle Royale",
    "Resolution": "2560x1440",
    "Avg_FPS": "70"

}

fps = answer_fps_query(user_input_fps, model)
print(f"FPS Inference: {fps}")
