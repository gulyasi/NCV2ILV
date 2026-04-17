from datasets import load_dataset
import os
import csv

def download_with_labels(limit=50):
    print("Loading dataset...")
    dataset = load_dataset("fhswf/german_handwriting", split="train")
    
    # Setup directories
    output_dir = "data/raw_handwriting"
    os.makedirs(output_dir, exist_ok=True)
    
    metadata_path = os.path.join("data", "metadata.csv")
    
    print(f"Saving {limit} images and creating metadata.csv...")

    with open(metadata_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write Header
        writer.writerow(["file_name", "label"])
        
        for i in range(limit):
            sample = dataset[i]
            img = sample['image']
            label_text = sample['text']
            
            # Create a clean, unique filename
            filename = f"handwriting_{i:04d}.png"
            img_path = os.path.join(output_dir, filename)
            
            # Save Image
            img.save(img_path)
            
            # Save Label to CSV
            writer.writerow([filename, label_text])
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{limit}")

    print(f"\nSuccess! Images are in: {output_dir}")
    print(f"Labels are recorded in: {metadata_path}")

if __name__ == "__main__":
    download_with_labels(100) # Increased to 100 for a better golden set
