
# coding: utf-8

# In[6]:

from skimage import data,io,filters
from os import listdir,system
import shutil
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy
from PIL import Image


# In[7]:

filepath="Manatee_dataset/"
infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"


# In[8]:

for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    #im_inv = numpy.invert(im)
    image_changed = image.replace("invert_","")
    outfilepath_changed = outfilepath+image_changed
    io.imsave(outfilepath_changed, im)


# In[9]:

filepath="Manatee_dataset/"
infilepath="Manatee_dataset/preprocessed_and_augmented_train/mean/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/mean/"


# In[10]:

for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    #im_inv = numpy.invert(im)
    image_changed = image.replace("mean_invert_","")
    outfilepath_changed = outfilepath+image_changed
    io.imsave(outfilepath_changed, im)


# In[11]:

filepath="Manatee_dataset/"
infilepath="Manatee_dataset/preprocessed_and_augmented_train/median/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/median/"


# In[12]:

for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    #im_inv = numpy.invert(im)
    image_changed = image.replace("median_invert_","")
    outfilepath_changed = outfilepath+image_changed
    io.imsave(outfilepath_changed, im)


# In[13]:

filepath="Manatee_dataset/"
infilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_strong/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_strong/"


# In[14]:

for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    #im_inv = numpy.invert(im)
    image_changed = image.replace("gaussian_strong_invert_","")
    outfilepath_changed = outfilepath+image_changed
    io.imsave(outfilepath_changed, im)


# In[15]:

filepath="Manatee_dataset/"
infilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_mild/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_mild/"


# In[16]:

for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    #im_inv = numpy.invert(im)
    image_changed = image.replace("gaussian_mild_invert_","")
    outfilepath_changed = outfilepath+image_changed
    io.imsave(outfilepath_changed, im)


# In[17]:

filepath="Manatee_dataset/"
infilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_real/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_real/"


# In[18]:

for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    #im_inv = numpy.invert(im)
    image_changed = image.replace("gabor_real_invert_","")
    outfilepath_changed = outfilepath+image_changed
    io.imsave(outfilepath_changed, im)


# In[19]:

filepath="Manatee_dataset/"
infilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_imaginary/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_imaginary/"


# In[20]:

for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    #im_inv = numpy.invert(im)
    image_changed = image.replace("gabor_imaginary_invert_","")
    outfilepath_changed = outfilepath+image_changed
    io.imsave(outfilepath_changed, im)


# In[ ]:



