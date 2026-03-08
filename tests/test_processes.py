from src.data.preprocess import preprocess_gpu_data
from src.models.train_model import train
from src.data.load_data import load_model
from src.models.predict import answer_fps_query

model = load_model()
user_input = {
    "gpu_name": "NVIDIA GeForce RTX 3080",
    "game": "Cyberpunk 2077",
    "resolution": "256"
}
fps = answer_fps_query(user_input, model)
print(f"FPS Inference: {fps}")
