import cv2
import numpy as np
import os
import pandas as pd

def segment_line_to_glyphs(image_path, label, writer_id, output_dir, row_idx):
    # 1. Load and check image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return 0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Denoising: Median blur is superior for salt-and-pepper noise
    # It replaces each pixel with the median of its neighbors
    denoised = cv2.medianBlur(gray, 5)

    # 3. Thresholding: Using Otsu to separate ink from paper
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Morphological Cleaning
    # Using a larger kernel (5x5) to remove larger noise clusters
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # DEBUG: Save this to your project root to see what the computer "sees"
    cv2.imwrite("debug_binary.png", binary)

    # 5. Segmentation: Finding connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    components = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        # INCREASED area threshold: 
        # Real handwritten letters are usually larger than 500 pixels at standard resolution.
        if area > 200 and h > 10 and w > 5: 
            components.append((x, y, w, h))
    
    # Sort left to right
    components.sort(key=lambda c: c[0])

    # 6. Save individual glyphs
    glyphs_saved = 0
    # Clean the label (remove spaces for mapping)
    clean_label = label.replace(" ", "")
    
    for i, (x, y, w, h) in enumerate(components):
        # We only save if we have a corresponding character in our label
        if i < len(clean_label):
            char = clean_label[i]
            
            # Add padding
            pad = 5
            y_s, y_e = max(0, y-pad), min(binary.shape[0], y+h+pad)
            x_s, x_e = max(0, x-pad), min(binary.shape[1], x+w+pad)
            
            glyph_crop = binary[y_s:y_e, x_s:x_e]
            
            # Save using character code and unique index
            char_code = ord(char)
            filename = f"char_{char_code}_{writer_id}_{row_idx}_{i}.png"
            cv2.imwrite(os.path.join(output_dir, filename), glyph_crop)
            glyphs_saved += 1
            
    return glyphs_saved

if __name__ == "__main__":
    RAW_DIR = "data/raw_handwriting"
    GLYPH_DIR = "data/glyphs"
    os.makedirs(GLYPH_DIR, exist_ok=True)
    
    metadata_path = "data/metadata.csv"
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
    else:
        metadata = pd.read_csv(metadata_path)
        print("Starting improved segmentation...")
        
        total_glyphs = 0
        # Testing on first 20 lines
        for idx, row in metadata.head(20).iterrows():
            img_path = os.path.join(RAW_DIR, row['file_name'])
            count = segment_line_to_glyphs(img_path, row['label'], "writer_german", GLYPH_DIR, idx)
            total_glyphs += count
            
        print(f"Finished. Saved {total_glyphs} glyphs to '{GLYPH_DIR}'.")
        print("Please inspect 'debug_binary.png' for thresholding quality.")
