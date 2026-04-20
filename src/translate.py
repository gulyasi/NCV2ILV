import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os

class QwenTranslator:
    def __init__(self, model_id="Qwen/Qwen3.5-0.8B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_id} onto {self.device}...")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        ).to(self.device)

    # WE MUST NAME THIS 'transcribe' TO MATCH YOUR PIPELINE
    def transcribe(self, image_path):
        if not os.path.exists(image_path):
            return "Error: Image not found."
            
        raw_image = Image.open(image_path).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Transcribe the German handwriting in this image. Output only the transcription."}
                ]
            },
        ]

        inputs = self.processor(
            text=self.processor.apply_chat_template(messages, add_generation_prompt=True),
            images=[raw_image],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=128)
            
        return self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
