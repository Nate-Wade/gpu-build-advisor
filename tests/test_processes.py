from src.data.preprocess import preprocess_gpu_data
import sys
sys.path.append("..")

data = preprocess_gpu_data()

print(data.head())
print(data.columns)
print(data.shape)
