
# coding: utf-8

# In[1]:

from skimage import data,io,filters
from os import listdir,system
import shutil
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy
from PIL import Image


# In[2]:

invert_mini_resized = []
invert_mini_resized_labels = []


# In[3]:

infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    #im = numpy.array(im)
    im = im.convert('L')
    invert_mini_resized.append(numpy.array(im))
    #invert_mini_resized.append(im)
    invert_mini_resized_labels.append(image[:-4])
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)


# In[4]:

invert_mini_resized_train_array = numpy.array(invert_mini_resized)
print invert_mini_resized_train_array.shape
invert_mini_resized_labels_array = numpy.array(invert_mini_resized_labels)
print invert_mini_resized_labels_array.shape
print invert_mini_resized_labels_array[0]


# In[5]:

mean_mini_resized = []
mean_mini_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/mean_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.convert('L')
    mean_mini_resized.append(numpy.array(im))
    mean_mini_resized_labels.append(image[:-4])
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
mean_mini_resized_train_array = numpy.array(mean_mini_resized)
print mean_mini_resized_train_array.shape    
mean_mini_resized_labels_array = numpy.array(mean_mini_resized_labels)
print mean_mini_resized_labels_array.shape
print mean_mini_resized_labels_array[0]


# In[6]:

median_mini_resized = []
median_mini_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/median_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.convert('L')
    median_mini_resized.append(numpy.array(im))
    median_mini_resized_labels.append(image[:-4])    
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
median_mini_resized_train_array = numpy.array(median_mini_resized)
print median_mini_resized_train_array.shape
median_mini_resized_labels_array = numpy.array(median_mini_resized_labels)
print median_mini_resized_labels_array.shape
print median_mini_resized_labels_array[0]


# In[7]:

test_invert_mini_resized = []
test_invert_mini_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/test_invert_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.convert('L')
    test_invert_mini_resized.append(numpy.array(im))
    test_invert_mini_resized_labels.append(image[:-6])       
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
test_invert_mini_resized_array = numpy.array(test_invert_mini_resized)
print test_invert_mini_resized_array.shape
test_invert_mini_resized_labels_array = numpy.array(test_invert_mini_resized_labels)
print test_invert_mini_resized_labels_array.shape
print test_invert_mini_resized_labels_array[0]


# In[8]:

gabor_imaginary_mini_resized = []
gabor_imaginary_mini_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_imaginary_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.convert('L')
    gabor_imaginary_mini_resized.append(numpy.array(im))
    gabor_imaginary_mini_resized_labels.append(image[:-4])    
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
gabor_imaginary_mini_resized_train_array = numpy.array(gabor_imaginary_mini_resized)
print gabor_imaginary_mini_resized_train_array.shape
gabor_imaginary_mini_resized_labels_array = numpy.array(gabor_imaginary_mini_resized_labels)
print gabor_imaginary_mini_resized_labels_array.shape
print gabor_imaginary_mini_resized_labels_array[0]


# In[9]:

gabor_real_mini_resized = []
gabor_real_mini_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_real_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.convert('L')
    gabor_real_mini_resized.append(numpy.array(im))
    gabor_real_mini_resized_labels.append(image[:-4])    
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
gabor_real_mini_resized_train_array = numpy.array(gabor_real_mini_resized)
print gabor_real_mini_resized_train_array.shape
gabor_real_mini_resized_labels_array = numpy.array(gabor_real_mini_resized_labels)
print gabor_real_mini_resized_labels_array.shape
print gabor_real_mini_resized_labels_array[0]


# In[10]:

gaussian_mild_mini_resized = []
gaussian_mild_mini_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_mild_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.convert('L')
    gaussian_mild_mini_resized.append(numpy.array(im))
    gaussian_mild_mini_resized_labels.append(image[:-4])    
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
gaussian_mild_mini_resized_train_array = numpy.array(gaussian_mild_mini_resized)
print gaussian_mild_mini_resized_train_array.shape
gaussian_mild_mini_resized_labels_array = numpy.array(gaussian_mild_mini_resized_labels)
print gaussian_mild_mini_resized_labels_array.shape
print gaussian_mild_mini_resized_labels_array[0]


# In[11]:

gaussian_strong_mini_resized = []
gaussian_strong_mini_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_strong_mini_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.convert('L')
    gaussian_strong_mini_resized.append(numpy.array(im))
    gaussian_strong_mini_resized_labels.append(image[:-4])    
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
gaussian_strong_mini_resized_train_array = numpy.array(gaussian_strong_mini_resized)
print gaussian_strong_mini_resized_train_array.shape
gaussian_strong_mini_resized_labels_array = numpy.array(gaussian_strong_mini_resized_labels)
print gaussian_strong_mini_resized_labels_array.shape
print gaussian_strong_mini_resized_labels_array[0]


# In[12]:

train_mini_dataset = []
train_mini_dataset.append(invert_mini_resized_train_array[0])
train_mini_dataset.append(mean_mini_resized_train_array[0])
train_mini_dataset.append(median_mini_resized_train_array[0])
train_mini_dataset.append(gabor_imaginary_mini_resized_train_array[0])
train_mini_dataset.append(gabor_real_mini_resized_train_array[0])
train_mini_dataset.append(gaussian_mild_mini_resized_train_array[0])
train_mini_dataset.append(gaussian_strong_mini_resized_train_array[0])
print numpy.array(train_mini_dataset).shape


# In[13]:

train_mini_dataset_labels = []
train_mini_dataset_labels.append(invert_mini_resized_labels_array[0])
train_mini_dataset_labels.append(mean_mini_resized_labels_array[0])
train_mini_dataset_labels.append(median_mini_resized_labels_array[0])
train_mini_dataset_labels.append(gabor_imaginary_mini_resized_labels_array[0])
train_mini_dataset_labels.append(gabor_real_mini_resized_labels_array[0])
train_mini_dataset_labels.append(gaussian_mild_mini_resized_labels_array[0])
train_mini_dataset_labels.append(gaussian_strong_mini_resized_labels_array[0])
print numpy.array(train_mini_dataset_labels)


# In[14]:

train_mini_dataset = []
train_mini_dataset_labels = []
for i in range(2291):
    train_mini_dataset.append(invert_mini_resized_train_array[i])
    train_mini_dataset.append(mean_mini_resized_train_array[i])
    train_mini_dataset.append(median_mini_resized_train_array[i])
    train_mini_dataset.append(gabor_imaginary_mini_resized_train_array[i])
    train_mini_dataset.append(gabor_real_mini_resized_train_array[i])
    train_mini_dataset.append(gaussian_mild_mini_resized_train_array[i])
    train_mini_dataset.append(gaussian_strong_mini_resized_train_array[i])
    train_mini_dataset_labels.append(invert_mini_resized_labels_array[i])
    train_mini_dataset_labels.append(mean_mini_resized_labels_array[i])
    train_mini_dataset_labels.append(median_mini_resized_labels_array[i])
    train_mini_dataset_labels.append(gabor_imaginary_mini_resized_labels_array[i])
    train_mini_dataset_labels.append(gabor_real_mini_resized_labels_array[i])
    train_mini_dataset_labels.append(gaussian_mild_mini_resized_labels_array[i])
    train_mini_dataset_labels.append(gaussian_strong_mini_resized_labels_array[i])
    


# In[15]:

print numpy.array(train_mini_dataset).shape


# In[16]:

print numpy.array(train_mini_dataset_labels).shape


# In[17]:

train_mini_dataset_array = numpy.array(train_mini_dataset)
# Write the array to disk
with file('Manatee_dataset/train_mini.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('# Array shape: {0}\n'.format(train_mini_dataset_array.shape))

    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for data_slice in train_mini_dataset_array:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        numpy.savetxt(outfile, data_slice, fmt='%-7.0f')

        # Writing out a break to indicate different slices...
        outfile.write('# New slice\n')


# In[18]:

# Read the array from disk
new_data = numpy.loadtxt('Manatee_dataset/train_mini.txt')

# Note that this returned a 2D array!
print new_data.shape

# However, going back to 3D is easy if we know the 
# original shape of the array
new_data = new_data.reshape((16037,25,54))

# Just to check that they're the same...
print numpy.all(new_data == train_mini_dataset_array)


# In[19]:

new_data.shape


# In[20]:

print numpy.all(new_data == train_mini_dataset_array)


# In[23]:

train_mini_labels_dataset_array = numpy.array(train_mini_dataset_labels)

train_mini_labels_dataset_array.tofile('Manatee_dataset/train_labels_mini.txt',sep='\n',format="%s")


# In[30]:

# Read the array from disk
new_data = numpy.loadtxt('Manatee_dataset/train_labels_mini.txt', dtype=str)

# Note that this returned an array!
print new_data.shape


# In[31]:

train_mini_labels_dataset_array.shape


# In[33]:

print numpy.all(new_data == train_mini_labels_dataset_array)


# In[34]:

test_mini_dataset_array = test_invert_mini_resized_array
test_mini_labels_dataset_array = test_invert_mini_resized_labels_array


# In[35]:


# Write the array to disk
with file('Manatee_dataset/test_mini.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('# Array shape: {0}\n'.format(test_mini_dataset_array.shape))

    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for data_slice in test_mini_dataset_array:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        numpy.savetxt(outfile, data_slice, fmt='%-7.0f')

        # Writing out a break to indicate different slices...
        outfile.write('# New slice\n')


# In[36]:

# Read the array from disk
new_data = numpy.loadtxt('Manatee_dataset/test_mini.txt')

# Note that this returned a 2D array!
print new_data.shape

# However, going back to 3D is easy if we know the 
# original shape of the array
new_data = new_data.reshape((122,25,54))

# Just to check that they're the same...
print numpy.all(new_data == test_mini_dataset_array)


# In[37]:

print new_data.shape


# In[39]:


test_mini_labels_dataset_array.tofile('Manatee_dataset/test_labels_mini.txt',sep='\n',format="%s")


# In[40]:

# Read the array from disk
new_data = numpy.loadtxt('Manatee_dataset/test_labels_mini.txt', dtype=str)

# Note that this returned an array!
print new_data.shape


# In[42]:

print numpy.all(new_data == test_mini_labels_dataset_array)


# In[2]:

invert_resized = []
invert_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/invert_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    #im = numpy.array(im)
    im = im.convert('L')
    invert_resized.append(numpy.array(im))
    #invert_resized.append(im)
    invert_resized_labels.append(image[:-4])
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
invert_resized_train_array = numpy.array(invert_resized)
print invert_resized_train_array.shape
invert_resized_labels_array = numpy.array(invert_resized_labels)
print invert_resized_labels_array.shape
print invert_resized_labels_array[0]


# In[3]:

mean_resized = []
mean_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/mean_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.convert('L')
    mean_resized.append(numpy.array(im))
    mean_resized_labels.append(image[:-4])
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
mean_resized_train_array = numpy.array(mean_resized)
print mean_resized_train_array.shape    
mean_resized_labels_array = numpy.array(mean_resized_labels)
print mean_resized_labels_array.shape
print mean_resized_labels_array[0]


# In[4]:

median_resized = []
median_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/median_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.convert('L')
    median_resized.append(numpy.array(im))
    median_resized_labels.append(image[:-4])    
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
median_resized_train_array = numpy.array(median_resized)
print median_resized_train_array.shape
median_resized_labels_array = numpy.array(median_resized_labels)
print median_resized_labels_array.shape
print median_resized_labels_array[0]


# In[5]:

test_invert_resized = []
test_invert_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/test_invert_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.convert('L')
    test_invert_resized.append(numpy.array(im))
    test_invert_resized_labels.append(image[:-6])       
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
test_invert_resized_array = numpy.array(test_invert_resized)
print test_invert_resized_array.shape
test_invert_resized_labels_array = numpy.array(test_invert_resized_labels)
print test_invert_resized_labels_array.shape
print test_invert_resized_labels_array[0]


# In[6]:

gabor_imaginary_resized = []
gabor_imaginary_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_imaginary_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.convert('L')
    gabor_imaginary_resized.append(numpy.array(im))
    gabor_imaginary_resized_labels.append(image[:-4])    
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
gabor_imaginary_resized_train_array = numpy.array(gabor_imaginary_resized)
print gabor_imaginary_resized_train_array.shape
gabor_imaginary_resized_labels_array = numpy.array(gabor_imaginary_resized_labels)
print gabor_imaginary_resized_labels_array.shape
print gabor_imaginary_resized_labels_array[0]


# In[7]:

gabor_real_resized = []
gabor_real_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/gabor_real_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.convert('L')
    gabor_real_resized.append(numpy.array(im))
    gabor_real_resized_labels.append(image[:-4])    
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
gabor_real_resized_train_array = numpy.array(gabor_real_resized)
print gabor_real_resized_train_array.shape
gabor_real_resized_labels_array = numpy.array(gabor_real_resized_labels)
print gabor_real_resized_labels_array.shape
print gabor_real_resized_labels_array[0]


# In[8]:

gaussian_mild_resized = []
gaussian_mild_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_mild_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.convert('L')
    gaussian_mild_resized.append(numpy.array(im))
    gaussian_mild_resized_labels.append(image[:-4])    
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
gaussian_mild_resized_train_array = numpy.array(gaussian_mild_resized)
print gaussian_mild_resized_train_array.shape
gaussian_mild_resized_labels_array = numpy.array(gaussian_mild_resized_labels)
print gaussian_mild_resized_labels_array.shape
print gaussian_mild_resized_labels_array[0]


# In[9]:

gaussian_strong_resized = []
gaussian_strong_resized_labels = []
infilepath="Manatee_dataset/preprocessed_and_augmented_train/gaussian_strong_resized/"
for image in listdir(infilepath):
    filename=infilepath+image
    im = Image.open(filename)
    im = im.convert('L')
    gaussian_strong_resized.append(numpy.array(im))
    gaussian_strong_resized_labels.append(image[:-4])    
    #im = im.resize((54, 25), Image.ANTIALIAS)
    #im.save(outfilepath+image)
gaussian_strong_resized_train_array = numpy.array(gaussian_strong_resized)
print gaussian_strong_resized_train_array.shape
gaussian_strong_resized_labels_array = numpy.array(gaussian_strong_resized_labels)
print gaussian_strong_resized_labels_array.shape
print gaussian_strong_resized_labels_array[0]


# In[10]:

train_dataset = []
train_dataset_labels = []
for i in range(2291):
    train_dataset.append(invert_resized_train_array[i])
    train_dataset.append(mean_resized_train_array[i])
    train_dataset.append(median_resized_train_array[i])
    train_dataset.append(gabor_imaginary_resized_train_array[i])
    train_dataset.append(gabor_real_resized_train_array[i])
    train_dataset.append(gaussian_mild_resized_train_array[i])
    train_dataset.append(gaussian_strong_resized_train_array[i])
    train_dataset_labels.append(invert_resized_labels_array[i])
    train_dataset_labels.append(mean_resized_labels_array[i])
    train_dataset_labels.append(median_resized_labels_array[i])
    train_dataset_labels.append(gabor_imaginary_resized_labels_array[i])
    train_dataset_labels.append(gabor_real_resized_labels_array[i])
    train_dataset_labels.append(gaussian_mild_resized_labels_array[i])
    train_dataset_labels.append(gaussian_strong_resized_labels_array[i])


# In[11]:

print numpy.array(train_dataset).shape
print numpy.array(train_dataset_labels).shape


# In[15]:

train_dataset_array = numpy.array(train_dataset)
# Write the array to disk
with file('Manatee_dataset/train.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('# Array shape: {0}\n'.format(train_dataset_array.shape))

    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    i=0
    for data_slice in train_dataset_array:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        numpy.savetxt(outfile, data_slice, fmt='%-7.0f')
        i=i+1
        if i%100==0:
            print i
        # Writing out a break to indicate different slices...
        outfile.write('# New slice\n')


# In[16]:

# Read the array from disk
new_data = numpy.loadtxt('Manatee_dataset/train.txt')

# Note that this returned a 2D array!
print new_data.shape

# However, going back to 3D is easy if we know the 
# original shape of the array
new_data = new_data.reshape((2291,250,540))

# Just to check that they're the same...
print numpy.all(new_data == train_dataset_array)
print new_data.shape


# In[17]:

train_labels_dataset_array = numpy.array(train_dataset_labels)

train_labels_dataset_array.tofile('Manatee_dataset/train_labels.txt',sep='\n',format="%s")


# In[18]:

# Read the array from disk
new_data = numpy.loadtxt('Manatee_dataset/train_labels.txt', dtype=str)

# Note that this returned an array!
print new_data.shape
print numpy.all(new_data == train_labels_dataset_array)


# In[19]:

test_dataset_array = test_invert_resized_array
test_labels_dataset_array = test_invert_resized_labels_array


# In[20]:

# Write the array to disk
with file('Manatee_dataset/test.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('# Array shape: {0}\n'.format(test_dataset_array.shape))

    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for data_slice in test_dataset_array:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        numpy.savetxt(outfile, data_slice, fmt='%-7.0f')

        # Writing out a break to indicate different slices...
        outfile.write('# New slice\n')


# In[21]:

# Read the array from disk
new_data = numpy.loadtxt('Manatee_dataset/test.txt')

# Note that this returned a 2D array!
print new_data.shape

# However, going back to 3D is easy if we know the 
# original shape of the array
new_data = new_data.reshape((122,250,540))

# Just to check that they're the same...
print numpy.all(new_data == test_dataset_array)
print new_data.shape


# In[22]:


test_labels_dataset_array.tofile('Manatee_dataset/test_labels.txt',sep='\n',format="%s")


# In[23]:

# Read the array from disk
new_data = numpy.loadtxt('Manatee_dataset/test_labels.txt', dtype=str)

# Note that this returned an array!
print new_data.shape


# In[ ]:



