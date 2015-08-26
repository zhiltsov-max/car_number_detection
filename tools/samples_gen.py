"""
This script loads an image, prints text on it and saves.
"""

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from random import randint, uniform

if (__name__ == "__main__"):
	# font = ImageFont.truetype(<font-file>, <font-size>)
	# Font files can be found at "C:/Windows/Fonts",
	# and the following function searches in it by default.
	# draw.text((x, y), "Text", (r, g, b), font=font)

	data = open("data.txt", "wb+")
	# Initialize symbol images
	font = ImageFont.truetype("RoadNumbers2.0.ttf", 16)
	symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "H", "K", "M", "O", "P", "T", "X", "Y"]

	symbolExamples = 100

	for i in xrange(0, len(symbols)):
		for idx in xrange(0, symbolExamples):
			font = ImageFont.truetype("RoadNumbers2.0.ttf", randint(12, 18))
			img = Image.new("1", (10, 10), (0))
			draw = ImageDraw.Draw(img)
			draw.text((uniform(-2, 4), uniform(-2, 4)), symbols[i], (randint(0, 255)), font = font)

			name = "p_" + str(i) + "_" + str(idx) + ".png"
			img.save(name, "PNG")
			data.write(name + " " + str(i) + "\r\n")

	data.close()