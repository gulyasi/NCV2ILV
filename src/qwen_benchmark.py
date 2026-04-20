import os
import torch
import pandas as pd
import difflib
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

class QwenEvaluator:
    def __init__(self, model_id="Qwen/Qwen3.5-0.8B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Warming up {model_id} on {self.device} ---")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )

    def transcribe(self, image_path):
        if not os.path.exists(image_path): return ""
        raw_image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe the German handwriting. Output only the text."}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[raw_image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=128)
        return self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

def run_benchmarks(limit=5):
    evaluator = QwenEvaluator()
    df = pd.read_csv("data/metadata.csv").head(limit)
    total_acc = 0
    for idx, row in df.iterrows():
        pred = evaluator.transcribe(os.path.join("data/raw_handwriting", row['file_name']))
        acc = difflib.SequenceMatcher(None, str(row['label']), pred).ratio()
        total_acc += acc
        print(f"File: {row['file_name']} | Acc: {acc:.2%} | Output: {pred}")
    print(f"\nFinal Average Accuracy: {total_acc/len(df):.2%}")

if __name__ == "__main__":
    run_benchmarks(5)
