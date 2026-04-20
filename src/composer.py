import json, random, os
from PIL import Image

def compose(text, output_name="handwritten_result.png"):
    with open('data/glyph_library.json', 'r') as f: lib = json.load(f)
    page = Image.new('L', (2480, 3508), 255)
    x, y, line_h = 150, 150, 200
    for char in text:
        if char == " ": x += 80; continue
        if char == "\n": x, y = 150, y + line_h; continue
        if char in lib:
            glyph = Image.open(random.choice(lib[char])).convert('L')
            if x + glyph.width > 2300: x, y = 150, y + line_h
            page.paste(Image.eval(glyph, lambda p: 255 - p), (x, y))
            x += glyph.width + 12
    page.save(output_name)
    print(f"File created: {output_name}")

if __name__ == "__main__":
    compose("this is a test.")
