import numpy as np
import pandas as pd
from sklearn import svm
import os
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from numpy import save
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array

images = os.listdir("D:/PDF/3rd year/Second term/machine learning/dog-cat classification/dataset")
# print(images)

print(type(images))
print(images[0])
im=images[0].replace("cat.", "cat")
print(im)
# for img in images:
    
#     if img.startswith('dog'):
#         print(img)
	# convert to numpy array
# 	photo = img_to_array(photo)
# img = imread(im)
# imshow(img)
# print(img.shape)

# resizedImage = resize(images,(128,64))

