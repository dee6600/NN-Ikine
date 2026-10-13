# Artificial Neural Network

# Importing the libraries
from itertools import dropwhile
import numpy as np
import pandas as pd
from sklearn import model_selection
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('train_rpp_rad.csv')

#select first 3 columns from dataset and save in X
X = dataset.iloc[:, :3].values

#select 4th, 5th and 6th column from dataset and save in y
y = dataset.iloc[:, 3:6].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


print(X_train.shape)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import numpy
from keras import backend as K
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import tensorflow as tf


model = Sequential()
model.add(Dense(32, input_dim=3, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation = 'linear'))
model.compile(loss='mean_absolute_error', optimizer = 'adam',metrics=['accuracy'])

#model.load_weights('model/model15000_rrr.h5')
model.fit(X_train,Y_train,epochs=8000,batch_size=768)

# evaluate the keras model
_, accuracy = model.evaluate(X_test, Y_test)
print('trainAccuracy: %.2f' % (accuracy*100))

_, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: %.2f' % (accuracy*100))

model.save('model/model8000_rpp_rad.h5')