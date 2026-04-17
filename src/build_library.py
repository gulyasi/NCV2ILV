import os
import json
from collections import defaultdict

def build_glyph_dictionary(glyph_dir, output_json='data/glyph_library.json'):
    # Dictionary mapping 'A' -> ['data/glyphs/char_65_...png']
    glyph_lib = defaultdict(list)
    
    if not os.path.exists(glyph_dir):
        print(f"Error: Directory {glyph_dir} does not exist.")
        return {}

    files = [f for f in os.listdir(glyph_dir) if f.endswith(".png")]
    
    for filename in files:
        try:
            # Filename: char_{code}_{writer}_{row}_{idx}.png
            parts = filename.split('_')
            if len(parts) < 2:
                continue
                
            char_code = int(parts[1])
            char = chr(char_code)
            
            # Store the full path relative to the project root
            full_path = os.path.join(glyph_dir, filename)
            glyph_lib[char].append(full_path)
        except (ValueError, IndexError):
            print(f"Skipping malformed filename: {filename}")
            continue
            
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
        
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(glyph_lib, f, indent=4, ensure_ascii=False)
        
    print(f"Library built with {len(glyph_lib)} unique characters.")
    print(f"Metadata saved to {output_json}")
    return glyph_lib

if __name__ == "__main__":
    # Point this to the folder where your new segmented glyphs are
    build_glyph_dictionary("data/glyphs")
