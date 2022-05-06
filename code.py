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
from matplotlib import pyplot
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import random as r
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier

images = os.listdir("D:/PDF/3rd year/Second term/machine learning/dog-cat-classification/dataset")
# print(images)

photos =[]
labels =[]
features = []

for img in images:
    label =0
    if img.startswith('dog'):
        label=1
    img = imread('dataset/'+img)
    #resize image
    resized_img = resize(img, (128,64))
    #generating HOG features
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), visualize=True, multichannel=True)
    features.append(fd)
    labels.append(label)



data = pd.DataFrame(features)
data['class'] = labels
print(data)
#before shuffle
#after shuffle
data = data.sample(frac=1).reset_index(drop=True)
print('-----------------------------------------------------------------------------')
print(data)

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

print(x)
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.0909)
print(x_test)

print(y_test)
sum = 0
#to know number of dogs in test data
for i in y_test:
    if i ==1:
        sum=sum+1
print(sum)



svm_kernel_ovo = OneVsOneClassifier(SVC(kernel='linear', C=0.01)).fit(x_train, y_train)
svm_kernel_ovr = OneVsRestClassifier(SVC(kernel='linear', C=0.01)).fit(x_train, y_train)

svm_linear_ovo = OneVsOneClassifier(LinearSVC(C=0.01),).fit(x_train, y_train)
svm_linear_ovr = OneVsRestClassifier(LinearSVC(C=0.01)).fit(x_train, y_train)




accuracy = svm_kernel_ovo.score(x_test, y_test)
print('Linear Kernel OneVsOne SVM accuracy for test data: ' + str(accuracy))

accuracy = svm_kernel_ovo.score(x_train, y_train)
print('Linear Kernel OneVsOne SVM accuracy for train data: ' + str(accuracy))

accuracy = svm_linear_ovo.score(x_test, y_test)
print('LinearSVC OneVsOne SVM accuracy for test data: ' + str(accuracy))

accuracy = svm_linear_ovo.score(x_train, y_train)
print('LinearSVC OneVsOne SVM accuracy for train data: ' + str(accuracy))
