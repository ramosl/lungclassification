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

drop_rate = 0.3

inputs = Input(shape=(64,64,64,1))

# 64x64x64
conv1a = Conv3D(32, [3,3,3], padding='same', activation= 'relu')(inputs)
conv1a = GaussianDropout(drop_rate)(conv1a)
conv1b = Conv3D(32, [3,3,3], padding='same', activation= 'relu')(conv1a)
conv1b = GaussianDropout(drop_rate)(conv1b)
pool1 = MaxPooling3D((2,2,2), strides = 2)(conv1b)

# 32x32x32
conv2a = Conv3D(64, [3,3,3], padding='same', activation= 'relu')(pool1)
conv2a = GaussianDropout(drop_rate)(conv2a)
conv2b = Conv3D(64, [3,3,3], padding='same', activation= 'relu')(conv2a)
conv2b = GaussianDropout(drop_rate)(conv2b)
pool2 = MaxPooling3D((2,2,2), strides = 2)(conv2b)

# 16x16x16
conv3a = Conv3D(128, [3,3,3], padding='same', activation= 'relu')(pool2)
conv3a = GaussianDropout(drop_rate)(conv3a)
conv3b = Conv3D(128, [3,3,3], padding='same', activation= 'relu')(conv3a)
conv3b = GaussianDropout(drop_rate)(conv3b)
pool3 = MaxPooling3D((2,2,2), strides = 2)(conv3b)

# 8x8x8
conv4a = Conv3D(128, [3,3,3], padding='same', activation= 'relu')(pool3)
conv4a = GaussianDropout(drop_rate)(conv4a)
conv4b = Conv3D(128, [3,3,3], padding='same', activation= 'relu')(conv4a)
conv4b = GaussianDropout(drop_rate)(conv4b)
upscale1 = UpSampling3D(size=(2, 2, 2))(conv4b)

concat1 = add([conv3b, upscale1])

# 16x16x16
conv5a = Conv3D(64, [3,3,3], padding='same', activation= 'relu')(concat1)
conv5a = GaussianDropout(drop_rate)(conv5a)
conv5b = Conv3D(64, [3,3,3], padding='same', activation= 'relu')(conv5a)
upscale2 = UpSampling3D(size=(2, 2, 2))(conv5b)

concat2 = add([conv2b, upscale2])

# 32x32x32
conv6a = Conv3D(32, [3,3,3], padding='same', activation= 'relu')(concat2)
conv6a = GaussianDropout(0.2)(conv6a)
conv6b = Conv3D(32, [3,3,3], padding='same', activation= 'relu')(conv6a)
conv6b = GaussianDropout(drop_rate)(conv6b)
upscale3 = UpSampling3D(size=(2, 2, 2))(conv6b)

concat3 = add([conv1b, upscale3])

# 64x64x64
x = BatchNormalization()(pool3)
x = Flatten()(x)
x = Dense(1024, activation = 'relu')(x)
x = GaussianDropout(drop_rate)(x)
predictions = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs = inputs, outputs = predictions)

adam = Adam(lr = 0.045)
model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer = adam)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(X, Y, epochs=20, batch_size=2, validation_split = 0.20, callbacks=[tensorboard], class_weight = {0:3, 1:7})

# Use test set to calculate final sensitivity and specificity
y_pred = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
area_under_curve = auc(fpr, tpr)
matrix = confusion_matrix(Y_test, y_pred)

print(matrix[0,0]) #tn
print(matrix[0,1]) #fn
print(matrix[1,0]) #fp
print(matrix[1,1]) #tp


total = matrix[0,0] + matrix[0,1] + matrix[1,0] + matrix[1,1]

true_pos = (matrix[1,1])/(matrix[1,1] + matrix[1,0]) #sensitivity
true_neg = matrix[0,0]/(matrix[0,0] + matrix[0,1]) #specificity
accuarcy = (matrix[0,0] + matrix[1,1])/total

print("Sensitivity is " + str(true_pos))
print("specificity is " + str(true_neg))
print("Accuracy is " + str(accuracy))


