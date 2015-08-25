"""
This script loads an image, prints text on it and saves.
"""

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


in_image = "lena.png"
out_image = "lena_lean.png"

img = Image.open(in_image)
draw = ImageDraw.Draw(img)
# font = ImageFont.truetype(<font-file>, <font-size>)
# Font files can be found at "C:/Windows/Fonts",
# and the following function searches in it by default.
font = ImageFont.truetype("arial.ttf", 72)

# draw.text((x, y), "Text", (r, g, b), font=font)
draw.text((0, 0), "Lena", (255, 255, 255), font=font)
img.save(out_image)
