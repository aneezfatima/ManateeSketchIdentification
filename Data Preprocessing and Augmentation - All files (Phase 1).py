
# coding: utf-8

# In[ ]:

from skimage import data,io,filters
from os import listdir,system
import shutil
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy
from PIL import Image


# In[ ]:

filepath="Manatee_dataset/"
infilepath="Manatee_dataset/sketches_train/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"


# In[ ]:

for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im_inv = numpy.invert(im)
    outfilepath_inverted = outfilepath+'invert_'+image
    io.imsave(outfilepath_inverted, im_inv)  


# In[ ]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)   
    if image.endswith('.tif'):  
        image = image[:-3]
        image = image+'jpg'
        io.imsave(outfilepath+image,im)  


# In[ ]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"
for image in listdir(infilepath):
    filename=infilepath+image    
    im = Image.open(filename)
    im = im.convert('L')
    im_inv = numpy.array(im)
    io.imsave(outfilepath+image,im_inv)


# In[ ]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_mild/"
for image in listdir(infilepath):   
    filename=infilepath+image
    im = Image.open(filename)   
    im_inv = numpy.array(im)
    im_gaussian_mild = filters.gaussian(im_inv, 0.5)
    outfilepath_gaussian_mild = outfilepath+'gaussian_mild_'+image
    io.imsave(outfilepath_gaussian_mild, im_gaussian_mild) 


# In[ ]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_strong/"
for image in listdir(infilepath):   
    filename=infilepath+image
    im = Image.open(filename)   
    im_inv = numpy.array(im)
    im_gaussian_strong = filters.gaussian(im_inv, 0.9)
    outfilepath_gaussian_strong = outfilepath+'gaussian_strong_'+image
    io.imsave(outfilepath_gaussian_strong, im_gaussian_strong) 


# In[ ]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_real/"
for image in listdir(infilepath):   
    filename=infilepath+image
    im = Image.open(filename)   
    im_inv = numpy.array(im)   
    im_gabor_real, im_gabor_imaginary = filters.gabor(im_inv, frequency=0.9)
    #print image, im_inv.shape
    outfilepath_gabor_real = outfilepath+'gabor_real_'+image
    io.imsave(outfilepath_gabor_real, im_gabor_real) 


# In[ ]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_imaginary/"
for image in listdir(infilepath):   
    filename=infilepath+image
    im = Image.open(filename)   
    im_inv = numpy.array(im)   
    im_gabor_real, im_gabor_imaginary = filters.gabor(im_inv, frequency=0.9)
    #print image, im_inv.shape
    outfilepath_gabor_imaginary = outfilepath+'gabor_imaginary_'+image
    io.imsave(outfilepath_gabor_imaginary, im_gabor_imaginary) 


# In[ ]:

from skimage.morphology import disk


# In[ ]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/median/"
for image in listdir(infilepath):   
    filename=infilepath+image
    im = Image.open(filename)   
    im_inv = numpy.array(im)   
    im_median = filters.median(im_inv,disk(2))
    outfilepath_median = outfilepath+'median_'+image
    io.imsave(outfilepath_median, im_median) 


# In[ ]:

from skimage.filters import rank


# In[ ]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/mean/"
for image in listdir(infilepath):   
    filename=infilepath+image
    im = Image.open(filename)   
    im_inv = numpy.array(im)   
    im_mean = rank.mean(im_inv,disk(2))
    outfilepath_mean = outfilepath+'mean_'+image
    io.imsave(outfilepath_mean, im_mean) 

