"""
This script loads an image, prints text on it and saves.
"""

import argparse
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

if (__name__ == "__main__"):
	parser = argparse.ArgumentParser(description="Extracts positive samples from a single image applying random geometric "
                                             "and photometric transforms. Trains a ANN classifier.")
	parser.add_argument("--width", type=int, default=10, help="width of the object image")
	parser.add_argument("--height", type=int, default=10, help="height of the object image")
	args = parser.parse_args()

	# font = ImageFont.truetype(<font-file>, <font-size>)
	# Font files can be found at "C:/Windows/Fonts",
	# and the following function searches in it by default.
	# draw.text((x, y), "Text", (r, g, b), font=font)

	# Initialize symbol images
	font = ImageFont.truetype("arial.ttf", 12)
	symbols = ["A", "B", "E", "K", "H", "M", "O", "P", "C", "T", "Y", "X", "D", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
	symbol_images = list()
	for i in xrange(0, len(symbols)):
		img = Image.new("1", (args.width, args.height), (0))
		draw = ImageDraw.Draw(img)
		draw.text((1, -2), symbols[i], (255), font = font)
		symbol_images.append(draw)	
		img.save("p_" + str(i) + ".png", "PNG")
		print "Image saved: ", "p_" + str(i) + ".png"
