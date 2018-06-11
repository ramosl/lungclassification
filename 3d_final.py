import numpy as np # linear algebra
from keras.models import Model, Sequential
from keras.layers import Conv3D, Input
from keras.layers import Dense
from keras.layers import MaxPooling3D
from keras.layers import BatchNormalization, Flatten, GaussianDropout, UpSampling3D, add
from keras.callbacks import TensorBoard
from time import time
from keras.optimizers import Adam
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix


x_train_raw = np.load('x_train.npy')
x_train_raw = np.expand_dims(x_train_raw, 4)
y_train_raw = np.load('stage1_yTrue.npy')
x_test_raw = np.load('x_test.npy')
x_test_raw = np.expand_dims(x_test_raw, 4)
y_test_raw = np.load('stage2_yTrue.npy')[:505]

X = np.concatenate((x_train_raw, x_test_raw))
Y = np.concatenate((y_train_raw, y_test_raw))

#create test set for final accuracy, specificity, and sensitivity check
X_test = X[X.shape[0]-101:X.shape[0]-1]
Y_test = Y[Y.shape[0]-101:Y.shape[0]-1]

#remove test set from normal training and validation
X = X[:X.shape[0]-101]
Y = Y[:Y.shape[0]-101]

print(X.shape)
print(X_test.shape)
print(Y.shape)
print(Y_test.shape)


drop_rate = 0.4

model = Sequential()

# 64x64x64
model.add(Conv3D(64, [3,3,3], padding='same', activation= 'relu', input_shape=(64, 64, 64,1)))
model.add(GaussianDropout(drop_rate))
model.add(Conv3D(64, [3,3,3], padding='same', activation= 'relu'))
model.add(Conv3D(64, [3,3,3], padding='same', activation= 'relu'))
model.add(Conv3D(64, [3,3,3], padding='same', activation= 'relu'))
model.add(Conv3D(64, [3,3,3], padding='same', activation= 'relu'))
model.add(Conv3D(64, [3,3,3], padding='same', activation= 'relu'))
model.add(MaxPooling3D((2,2,2), strides = 2))

# 32x32x32
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(GaussianDropout(drop_rate))
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(MaxPooling3D((2,2,2), strides = 2))

#16x16x16
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(GaussianDropout(drop_rate))
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(MaxPooling3D((2,2,2), strides = 2))

# 8x8x8
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(GaussianDropout(drop_rate))
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(Conv3D(128, [3,3,3], padding='same', activation= 'relu'))
model.add(MaxPooling3D((2,2,2), strides = 2))

# 4x4x4
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(GaussianDropout(drop_rate))
model.add(Dense(1, activation = 'sigmoid'))


adam = Adam(lr = 0.003, decay=1e-6, beta_1=0.99, beta_2 = 0.999 )
model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer = adam)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(X, Y, epochs=5, batch_size=5, validation_split = 0.20, callbacks=[tensorboard], class_weight = {0:(1/7), 1:(1/3)})


# Use test set to calculate final sensitivity and specificity
y_pred = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
area_under_curve = auc(fpr, tpr)
matrix = confusion_matrix(Y_test, y_pred.astype(int))

total = matrix[0,0] + matrix[0,1] + matrix[1,0] + matrix[1,1]

true_pos = (matrix[1,1])/(matrix[1,1] + matrix[1,0]) #sensitivity
true_neg = matrix[0,0]/(matrix[0,0] + matrix[0,1]) #specificity
accuracy = (matrix[0,0] + matrix[1,1])/total

print(matrix[0,0]) #tn
print(matrix[0,1]) #fn
print(matrix[1,0]) #fp
print(matrix[1,1]) #tp

print("Sensitivity is " + str(true_pos))
print("specificity is " + str(true_neg))
print("Accuracy is " + str(accuracy))


