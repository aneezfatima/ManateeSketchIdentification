
# coding: utf-8

# In[1]:

from skimage import data,io,filters
from os import listdir,system
import shutil
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy
from PIL import Image


# In[7]:

infilepath="Manatee_dataset/sketches_train/"


# In[8]:

img_dim = []
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    img_dim.append(im.size)
    #im_inv = numpy.invert(im)
    #image_changed = image.replace("invert_","")
    #outfilepath_changed = outfilepath+image_changed
    #io.imsave(outfilepath_changed, im)
print img_dim    


# In[11]:

print min(img_dim)


# In[23]:

length = []
for i in img_dim:
    length.append(i[0])
print length    


# In[28]:

def avg(list):
    sum = 0
    for elm in list:
        sum += elm
    return str(sum/(len(list)*1.0))


# In[29]:

print avg(length)


# In[31]:

breadth = []
for i in img_dim:
    breadth.append(i[1])


# In[32]:

print avg(breadth)


# In[40]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/invert_resized/"

from resizeimage import resizeimage

def resize_file(in_file, out_file, size):
    with open(in_file) as fd:
        image = resizeimage.resize_thumbnail(Image.open(fd), size)
    image.save(outfilepath+out_file)
    image.close()


# In[41]:

for image in listdir(infilepath):
    filename=infilepath+image
    resize_file(filename, image, (540, 250))
    #im = Image.open(filename)
    #img_dim.append(im.size)


# In[46]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/invert_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((540, 250), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[47]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert_resized/"
img_dim = []
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    img_dim.append(im.size)
    #im_inv = numpy.invert(im)
    #image_changed = image.replace("invert_","")
    #outfilepath_changed = outfilepath+image_changed
    #io.imsave(outfilepath_changed, im)
print img_dim 


# In[84]:

min(img_dim)


# In[52]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/mean/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/mean_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((540, 250), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[53]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/mean_resized/"
img_dim = []
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    img_dim.append(im.size)


# In[56]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/median/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/median_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((540, 250), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[57]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/median_resized/"
img_dim = []
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    img_dim.append(im.size)


# In[61]:


infilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_strong/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_strong_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((540, 250), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[62]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_strong_resized/"
img_dim = []
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    img_dim.append(im.size)


# In[65]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_mild/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_mild_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((540, 250), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[66]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_mild_resized/"
img_dim = []
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    img_dim.append(im.size)


# In[69]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_real/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_real_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((540, 250), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[70]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_real_resized/"
img_dim = []
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    img_dim.append(im.size)


# In[73]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_imaginary/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_imaginary_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((540, 250), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[74]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_imaginary_resized/"
img_dim = []
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    img_dim.append(im.size)


# In[78]:

from skimage import data,io,filters
infilepath="Manatee_dataset/sketches_test/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/test_invert/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    #im = im.resize((540, 250), Image.ANTIALIAS)
    im_inv = numpy.invert(im)
    io.imsave(outfilepath+image,im_inv)
    #im.save(outfilepath+image)


# In[79]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/test_invert/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/test_invert/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)   
    if image.endswith('.tif'):  
        image = image[:-3]
        image = image+'jpg'
        io.imsave(outfilepath+image,im)  


# In[80]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/test_invert/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/test_invert/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    #im = im.resize((540, 250), Image.ANTIALIAS)
    im_inv = numpy.invert(im)
    io.imsave(outfilepath+image,im_inv)
    #im.save(outfilepath+image)


# In[81]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/test_invert/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/test_invert_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((540, 250), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[82]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/test_invert_resized/"
img_dim = []
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    img_dim.append(im.size)


# In[86]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert_resized/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/invert_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((54, 25), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[87]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/mean_resized/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/mean_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((54, 25), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[88]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/median_resized/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/median_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((54, 25), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[90]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_strong_resized/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_strong_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((54, 25), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[91]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_mild_resized/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_mild_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((54, 25), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[92]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_real_resized/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_real_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((54, 25), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[93]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_imaginary_resized/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_imaginary_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((54, 25), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[94]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/test_invert_resized/"
outfilepath="Manatee_dataset/preprocessed_and_augmented_train/test_invert_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.resize((54, 25), Image.ANTIALIAS)
    im.save(outfilepath+image)


# In[ ]:



