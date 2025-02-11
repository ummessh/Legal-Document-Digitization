from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import os

# Define the path to save the model
model_path = "models/mt5-correction"

# Create the directory if it doesn't exist
os.makedirs(model_path, exist_ok=True)

# Download and save the model and tokenizer
print("Downloading mT5 model...")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

print(f"Saving model and tokenizer to {model_path}...")
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print("Model and tokenizer saved successfully!")
