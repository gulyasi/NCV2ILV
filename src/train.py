import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image

class SimpleBaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleBaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GlyphDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        char_codes = sorted(list(set([f.split('_')[1] for f in self.image_files])))
        self.char_to_idx = {code: idx for idx, code in enumerate(char_codes)}

    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert('L')
        label = self.char_to_idx[fname.split('_')[1]]
        if self.transform: img = self.transform(img)
        return img, label

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = transforms.Compose([
        transforms.Resize((64,64)), 
        transforms.RandomAffine(10, translate=(0.1, 0.1)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5,),(0.5,))
    ])
    ds = GlyphDataset("data/glyphs", transform=t)
    train_ds, val_ds = random_split(ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))])
    loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    
    model = SimpleBaselineCNN(len(ds.char_to_idx)).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    
    for epoch in range(15):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad(); crit(model(imgs), lbls).backward(); opt.step()
        print(f"Epoch {epoch+1} complete")
    torch.save(model.state_dict(), "data/baseline_model.pth")



''''
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-0.8B")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
'''