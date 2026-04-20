import os
import torch
from translate import QwenTranslator

def run_translation_only(input_image_path):
    print(f"--- Processing: {os.path.basename(input_image_path)} ---")
    
    # 1. Initialize the Brain (Only once)
    # This loads the 0.8B model into your GPU/CPU
    translator = QwenTranslator()
    
    # 2. Perform the Transcription
    # Qwen looks at the pixels and returns a Python String
    transcription = translator.transcribe(input_image_path)
    
    # 3. Output to Terminal
    print("\n" + "="*30)
    print("HANDWRITING RECOGNITION OUTPUT:")
    print("-" * 30)
    print(transcription)
    print("="*30 + "\n")

if __name__ == "__main__":
    # Path to your handwriting image
    IMAGE_PATH = "data/raw_handwriting/long_handwritten.png"
    
    if os.path.exists(IMAGE_PATH):
        run_translation_only(IMAGE_PATH)
    else:
        print(f"Error: Could not find image at {IMAGE_PATH}")
