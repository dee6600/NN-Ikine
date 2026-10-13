
#print python version
import sys
print(sys.version)


# Importing the libraries
import numpy as np
print('numpy version: ', np.__version__)
from numpy.lib.histograms import _ravel_and_check_weights
import pandas as pd
from sklearn import model_selection
import keras


read_data = pd.read_csv('dataset/EE_pose_rrr.csv', header=None)

#data frame data to np array
data = read_data.values

data = np.array(data)


# #custom input for prediction
data = np.array([[1,4,9]])

#load the model
model = keras.models.load_model('model/model20000_rrr.h5')

predictions = model.predict(data)

print(predictions)
q1,q2,q3 = predictions[0][0],predictions[0][1],predictions[0][2]
q1 = float("{:.4f}".format(q1))
q2 = float("{:.4f}".format(q2))
q3 = float("{:.4f}".format(q3))

print(q1,q2,q3)
pred = [[q1,q2,q3]]
# predictions to csv with pandas
df = pd.DataFrame(pred)
df.to_csv('dataset/JS_pos_rrr.csv', index=False, header=False)

