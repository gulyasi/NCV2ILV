import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from train import GlyphDataset, HandwritingClassifier # Reusing your classes

def evaluate_model():
    # Settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GLYPH_DIR = "data/glyphs"
    MODEL_PATH = "data/handwriting_model.pth"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Train the model first!")
        return

    # 1. Prepare Data (Same transforms as training)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = GlyphDataset(img_dir=GLYPH_DIR, transform=transform)
    # We'll just use a portion for testing
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 2. Load Model
    num_classes = len(dataset.char_to_idx)
    model = HandwritingClassifier(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Evaluating model on {len(dataset)} glyphs...")

    correct = 0
    total = 0
    
    # 3. Inference
    with torch.no_grad():
        # Let's look at the first 20 predictions specifically
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i < 10:
                actual_char = dataset.idx_to_char[labels.item()]
                pred_char = dataset.idx_to_char[predicted.item()]
                print(f"Sample {i}: Actual='{actual_char}', Predicted='{pred_char}'")

    accuracy = 100 * correct / total
    print(f"\nFinal Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()
