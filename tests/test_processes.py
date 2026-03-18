from src.data.preprocess import preprocess_gpu_data
from src.models.train_model import train
from src.data.load_data import load_model
from src.inference.router import answer_fps_query

model = load_model()

user_input_easy = {"gpu_name": "RTX 3080", "Game_Name": "Cyberpunk 2077",
                   "Resolution": "1440p", "Setting": "Ultra"}

user_input_gpu_name = {"Game_Name": "warzone",
                       "fps": 120, "Resolution": "1080", "Setting": "high", "launch_price_USD": 700}


fps = answer_fps_query(user_input_gpu_name, model)
print(f"FPS Inference: {fps}")
