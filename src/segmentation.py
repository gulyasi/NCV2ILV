import cv2
import numpy as np
import os
import pandas as pd

def extract_glyphs(image_path, label, output_dir, row_idx):
    img = cv2.imread(image_path)
    if img is None: return
    # Blue Mask logic to delete grey grid lines
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    components = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 100: components.append((x, y, w, h))
    
    components.sort(key=lambda c: c[0])
    clean_label = label.replace(" ", "")
    
    for i, (x, y, w, h) in enumerate(components):
        if i < len(clean_label):
            char_code = ord(clean_label[i])
            cv2.imwrite(os.path.join(output_dir, f"char_{char_code}_{row_idx}_{i}.png"), binary[y:y+h, x:x+w])

if __name__ == "__main__":
    os.makedirs("data/glyphs", exist_ok=True)
    df = pd.read_csv("data/metadata.csv")
    for idx, row in df.iterrows():
        extract_glyphs(f"data/raw_handwriting/{row['file_name']}", row['label'], "data/glyphs", idx)
