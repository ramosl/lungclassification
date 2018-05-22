
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import os
import os.path
import tensorflow as tf

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology


# Some constants 
INPUT_FOLDER = './stage2/'
OUTPUT_FOLDER = './processed_images_stage_2/'
patients = os.listdir(INPUT_FOLDER)


labels = pd.read_csv('stage2_solution.csv')
print(len(labels))


patients.sort()
print(len(patients))


# In[ ]:


# Load the scans in given folder path
#
# Credit to Guido Zuidhof from the Full Preprocessing Tutorial kernal
# from the Kaggle Data Science Bowl 2017 competition for this code snippet. 
def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


# In[ ]:


# Stack all the slices to create an image of a lung
# Convert the pixel values to Hounsfield Units (HU)
#
# Credit to Guido Zuidhof from the Full Preprocessing Tutorial kernal
# from the Kaggle Data Science Bowl 2017 competition for this code snippet. 
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


# In[ ]:


#Resample all the slices such that the spacing between them is 1 mm x 1mm x 1mm
#
# Credit to Guido Zuidhof from the Full Preprocessing Tutorial kernal
# from the Kaggle Data Science Bowl 2017 competition for this code snippet.
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing


# In[ ]:


# For every axial slice in the scan, determine the largest solid connected component 
# (the body+air around the person), and set others to 0. This fills the structures in the lungs in the mask.
#
# Credit to Guido Zuidhof from the Full Preprocessing Tutorial kernal
# from the Kaggle Data Science Bowl 2017 competition for this code snippet.
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

#segments out lung tissue from the rest of the CT scan
#
# Credit to Guido Zuidhof from the Full Preprocessing Tutorial kernal
# from the Kaggle Data Science Bowl 2017 competition for this code snippet.
def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image  


# In[ ]:


# plot the given 3d image
# Not used for actual preprocessing except to check correctness of methods
#
# Credit to Guido Zuidhof from the Full Preprocessing Tutorial kernal
# from the Kaggle Data Science Bowl 2017 competition for this code snippet.
def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces, x, y = measure.marching_cubes(p, threshold) 

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


# In[ ]:


#Method to zero-center images. PIXEL_Mean value is based on LUNA16 dataset, which is similar to this dataset.
#
# Credit to Guido Zuidhof from the Full Preprocessing Tutorial kernal
# from the Kaggle Data Science Bowl 2017 competition for this code snippet.
PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image


# In[ ]:


#Method to normalize the image data since HU over +400 is uninteresting since
#it is simply bone.
#
# Credit to Guido Zuidhof from the Full Preprocessing Tutorial kernal
# from the Kaggle Data Science Bowl 2017 competition for this code snippet.

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


# In[ ]:


#Preprocess all the images for the patients in a given stage. 
for i in range(1, len(patients)):
    
    if patients[i] == '.DS_Store': continue;
    print(str(i))
    patient = load_scan(INPUT_FOLDER + patients[i])
    patient_pixels = get_pixels_hu(patient)
    pix_resampled, spacing = resample(patient_pixels, patient, [1,1,1])
    segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    
    #comment the following line out when actually preprocessing the entire dataset.
    #plot_3d(segmented_lungs_fill,0) 
    
    np.save(OUTPUT_FOLDER + 'patient_' + str(i), segmented_lungs_fill)
 
    


# In[ ]:


#Create a np vector of the true values based on patient numbers.
# 1 = cancer; 0 = no cancer
true_cancer_labels = np.zeros((len(patients), 1))

for i in range(1, len(patients)):
    path = OUTPUT_FOLDER + 'patient_' + str(i) + '.npy'
    if os.path.isfile(path):
        true_cancer_labels[i] = labels.loc[labels['id'] == patients[i]].values[0][1]
        true_cancer_labels[i]
        
np.save('stage2_yTrue', true_cancer_labels)

       


# In[ ]:


#Method that performs and the normalization and zeroing of data in one function.
#Covers all patients in a stage.
def normalize_and_zero():
    for i in range(0, len(patients)):
        path = OUTPUT_FOLDER + 'patient_' + str(i) + '.npy'
        if os.path.isfile(path):
            image = np.load(path)
            image = normalize(image)
            image = zero_center(image)
            np.save(OUTPUT_FOLDER + 'patient_' + str(i), image)
        
        


# In[ ]:


#Returns the tuple needed to pad a current sized dimension to the desired sized dimension using np.pad
def getPadding(current, desired):
    after = (desired - current)/2
    before = after
    if current %2 != 0:
        before += 1  #arbitrarily decide to divide odd numbers to favor an extra pixel on the left.
    return (before, after)


# In[ ]:


#Pad a given image to the given sizes for each dimension.
def padImage(image,size_x, size_y, size_z):
    image = np.pad(image, (getPadding(image.shape[0], size_x), getPadding(image.shape[1], size_y), getPadding(image.shape[2], size_z)), mode = 'edge')
    return image


# In[ ]:


#Crop the center an image to a desired size. 
def crop(image, crop_size):
    
    start_x = image.shape[0]/2 - crop_size/2
    end_x = image.shape[0]/2 + crop_size/2
    start_y = image.shape[1]/2 - crop_size/2
    end_y = image.shape[1]/2 + crop_size/2
    start_z = image.shape[2]/2 - crop_size/2
    end_z = image.shape[2]/2 + crop_size/2
    process_image = image[start_x:end_x, start_y:end_y, start_z:end_z]
    return process_image


# In[ ]:


#find the max axis sizes to determine the padding to normalize images without resizing them. 
max_x = -1;
max_y = -1;
max_z = -1
for i in range(0, len(patients)):
        path = OUTPUT_FOLDER + 'patient_' + str(i) + '.npy'
        if os.path.isfile(path):
            image = np.load(path)
            if image.shape[0] > max_x:
                max_x = image.shape[0]
            if image.shape[1] > max_y:
                max_y = image.shape[1]
            if image.shape[2] > max_z:
                max_z = image.shape[2]
print("max x is " + str(max_x))
print("max y is " + str(max_y))
print("max z is " + str(max_z))


# In[ ]:


#Max axis sizes found by the previous block
x_size = 404
y_size = 500
z_size = 500

height_width = 64     #desired height and width dimension to be used for resizing
batch_size = 20       #Number of examples to group together in one mini_train/mini_test file

batch_num = 0    #used to keep track of which batch we are on
count = 0        #used to keep track of which example in a given batch we are in.

#create training set of size 100
x_train = np.zeros((batch_size, height_width,height_width,z_size))
with tf.Session() as sess:
    for i in range(1, 101): #patient 1 is first 
        #reset x_train for next batch
        if count == 0:
            x_train = np.zeros((batch_size, height_width,height_width,z_size))  #reset the x_train array at the start of a batch
        
        path = OUTPUT_FOLDER + 'patient_' + str(i) + '.npy'
        if os.path.isfile(path):
            print(str(i))
            image = np.load(path)
            image = tf.image.resize_images(image, (height_width,height_width), method = tf.image.ResizeMethod.AREA).eval()
            image = padImage(image, height_width,height_width,500)  #Ensure that all the dimensions of the resized images match
            x_train[count] = image  #add preprocessed example to train_set
            
            #On the 20th example, save the mini_batch and reset variables as necessary
            if count % 19 == 0 and count != 0:
                print(x_train.shape)
                np.save('./database/mini_train_x_' + str(batch_num), x_train)
                count = 0
                batch_num += 1
            else:    
                count += 1


#Repeat the above for the test set of size 100
batch_num = 0
count = 0
x_test = np.zeros((batch_size, height_width,height_width,z_size))

with tf.Session() as sess:
    for i in range(101, 201):
        #reset x_train for next batch
        if count == 0:
            x_test = np.zeros((batch_size, height_width,height_width,z_size))
            
        path = OUTPUT_FOLDER + 'patient_' + str(i) + '.npy'
        if os.path.isfile(path):
            print(str(i))
            image = np.load(path)
            image = tf.image.resize_images(image, (height_width,height_width), method = tf.image.ResizeMethod.AREA).eval()
            image = padImage(image, height_width,height_width,500)
            
            x_test[count] = image
            if count % 19 == 0 and count != 0:
                print(x_test.shape)
                np.save('./database/mini_test_x_' + str(batch_num), x_test)
                count = 0
                batch_num += 1
            else:    
                count += 1
            


# In[ ]:


y_train = np.load('stage_2yTrue.npy')[0:100]   #Create train set true values
y_test = np.load('stage_2yTrue.npy')[100:200]  #Create test set true values
print(y_train.shape)
print(y_test.shape)
np.save('mini_y_train', y_train)   #Save the train set true values
np.save('mini_y_test', y_test)     #Save the test set true values

