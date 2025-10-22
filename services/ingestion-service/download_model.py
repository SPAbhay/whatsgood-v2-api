from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_PATH = './model'

# Create the directory if it doesn't exist
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Download and save the model
print(f"Downloading model '{MODEL_NAME}' to '{MODEL_PATH}'...")
model = SentenceTransformer(MODEL_NAME)
model.save(MODEL_PATH)
print("Model download complete.")