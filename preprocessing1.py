from PIL import Image
from os import listdir
import shutil

filepath="A:/MS Courses/Big Data Ecosystems/Project/Dataset/Partitioned/sketches_train/"

for image in listdir(filepath):
	filename=filepath+image
	img=Image.open(filename)
  #Add only the images with fixed manatee outline to the preprocessed folders
	if img.size==(559,259):
		shutil.copy2(filename, 'A:/MS Courses/Big Data Ecosystems/Project/Dataset/Preprocessed1/train/')
