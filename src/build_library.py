import os, json
from collections import defaultdict

def build_lib():
    lib = defaultdict(list)
    for f in os.listdir("data/glyphs"):
        if f.endswith(".png"):
            char = chr(int(f.split('_')[1]))
            lib[char].append(os.path.join("data/glyphs", f))
    with open('data/glyph_library.json', 'w', encoding='utf-8') as f:
        json.dump(lib, f, indent=4, ensure_ascii=False)
    print(f"Library built with {len(lib)} unique glyph types.")

if __name__ == "__main__":
    build_lib()
