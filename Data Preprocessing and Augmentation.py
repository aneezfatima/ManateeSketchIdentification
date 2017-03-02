
# coding: utf-8

# In[1]:

from skimage import data, io, filters


# In[2]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[3]:

im = io.imread("Downloads/U1319_B.jpg")


# In[4]:

plt.imshow(im, cmap='gray', interpolation='nearest')


# In[5]:

import numpy
im_inv = numpy.invert(im)


# In[6]:

plt.imshow(im_inv, cmap='gray', interpolation='nearest')


# In[7]:

im_gaussian_mild = filters.gaussian(im_inv, 0.4)


# In[8]:

plt.imshow(im_gaussian_mild, cmap='gray', interpolation='nearest')


# In[9]:

im_gaussian_strong = filters.gaussian(im_inv, 1.0)


# In[10]:

plt.imshow(im_gaussian_strong, cmap='gray', interpolation='nearest')


# In[11]:

im_gabor_real_less_sensitive, im_gabor_imaginary_less_sensitive = filters.gabor(im_inv, frequency=0.6)


# In[12]:

plt.imshow(im_gabor_real_less_sensitive, cmap='gray', interpolation='nearest')


# In[13]:

plt.imshow(im_gabor_imaginary_less_sensitive, cmap='gray', interpolation='nearest')


# In[14]:

im_gabor_real_more_sensitive, im_gabor_imaginary_more_sensitive = filters.gabor(im_inv, frequency=0.9)


# In[15]:

plt.imshow(im_gabor_real_more_sensitive, cmap='gray', interpolation='nearest')


# In[16]:

plt.imshow(im_gabor_imaginary_more_sensitive, cmap='gray', interpolation='nearest')


# In[17]:

im2 = io.imread("Downloads/U1523_C.jpg")
plt.imshow(im2, cmap='gray', interpolation='nearest')


# In[18]:

import numpy
im2_inv = numpy.invert(im2)
plt.imshow(im2_inv, cmap='gray', interpolation='nearest')


# In[19]:

im2_edges_laplace = filters.laplace(im2_inv)
plt.imshow(im2_edges_laplace , cmap='gray', interpolation='nearest')


# In[20]:

im2_vertical_edges_prewitt = filters.prewitt_v(im2_inv)
plt.imshow(im2_vertical_edges_prewitt , cmap='gray', interpolation='nearest')


# In[21]:

im2_vertical_edges_scharrv = filters.scharr_v(im2_inv)
plt.imshow(im2_vertical_edges_scharrv , cmap='gray', interpolation='nearest')


# In[22]:

im2_vertical_edges_sobelv = filters.sobel_v(im2_inv)
plt.imshow(im2_vertical_edges_sobelv , cmap='gray', interpolation='nearest')


# In[23]:

im2_threshold_adaptive_mean = filters.threshold_adaptive(im2_inv, 25, 'mean')
plt.imshow(im2_threshold_adaptive_mean , cmap='gray', interpolation='nearest')


# In[24]:

im2_threshold_adaptive_median = filters.threshold_adaptive(im2_inv, 35, 'median')
plt.imshow(im2_threshold_adaptive_median , cmap='gray', interpolation='nearest')


# In[25]:

from skimage.morphology import disk
im2_median = filters.median(im2_inv,disk(2))
plt.imshow(im2_median , cmap='gray', interpolation='nearest')


# In[26]:

im2_roberts = filters.roberts(im2_inv)
plt.imshow(im2_roberts , cmap='gray', interpolation='nearest')


# In[27]:

im2_roberts_neg_diag = filters.roberts_neg_diag(im2_inv)
plt.imshow(im2_roberts_neg_diag , cmap='gray', interpolation='nearest')


# In[28]:

im2_roberts_pos_diag = filters.roberts_pos_diag(im2_inv)
plt.imshow(im2_roberts_pos_diag , cmap='gray', interpolation='nearest')


# In[29]:

from skimage.filters import rank
im2_mean = rank.mean(im2_inv,disk(2))
plt.imshow(im2_mean , cmap='gray', interpolation='nearest')

