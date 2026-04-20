import os
import pandas as pd
from translate import QwenTranslator # Import the Class, not the function

def run_benchmarks(limit=5):
    # 1. Initialize once
    translator = QwenTranslator()
    
    # 2. Load metadata
    df = pd.read_csv("data/metadata.csv").head(limit)
    
    print(f"\nStarting Benchmarks (Limit: {limit})...\n")
    
    for idx, row in df.iterrows():
        path = os.path.join("data/raw_handwriting", row['file_name'])
        
        # 3. Fast inference (using the warm model)
        prediction = translator.transcribe(path)
        
        print(f"[{idx+1}/{limit}] File: {row['file_name']}")
        print(f"  Actual: {row['label']}")
        print(f"  Qwen:   {prediction}")
        print("-" * 30)

if __name__ == "__main__":
    run_benchmarks(limit=10) # Set your desired limit here
