import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import os
from PIL import Image

# 1. The Dataset Class
# This now looks directly at your individual glyph files
class GlyphDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        
        # Extract unique characters to create a label mapping
        # Filename: char_{code}_{writer}_{row}_{idx}.png
        char_codes = sorted(list(set([f.split('_')[1] for f in self.image_files])))
        self.char_to_idx = {code: idx for idx, code in enumerate(char_codes)}
        self.idx_to_char = {idx: chr(int(code)) for code, idx in self.char_to_idx.items()}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.img_dir, filename)
        
        # Load image and convert to Grayscale ('L')
        image = Image.open(img_path).convert('L')
        
        # Get label from filename
        char_code = filename.split('_')[1]
        label_idx = self.char_to_idx[char_code]

        if self.transform:
            image = self.transform(image)

        return image, label_idx

# 2. ResNet-18 Architecture
class HandwritingClassifier(nn.Module):
    def __init__(self, num_classes):
        super(HandwritingClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Modify for 1-channel grayscale input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 3. Training Function
def run_training():
    # Settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GLYPH_DIR = "data/glyphs"
    BATCH_SIZE = 32
    EPOCHS = 15

    # Data Preparation
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = GlyphDataset(img_dir=GLYPH_DIR, transform=transform)
    num_classes = len(full_dataset.char_to_idx)
    
    # Split into Train/Validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model Setup
    model = HandwritingClassifier(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training on {DEVICE} with {num_classes} character classes.")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

    # Save the weights
    torch.save(model.state_dict(), "data/handwriting_model.pth")
    print("Training complete. Model saved to data/handwriting_model.pth")

if __name__ == "__main__":
    if os.path.exists("data/glyphs") and len(os.listdir("data/glyphs")) > 0:
        run_training()
    else:
        print("Error: No glyphs found in data/glyphs. Run segmentation.py first.")
