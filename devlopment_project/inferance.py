# Importing the libraries
import numpy as np
import pandas as pd
from sklearn import model_selection
import keras


# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('train8000test.csv')

#select first 3 columns from dataset and save in X
X = dataset.iloc[:, :3].values

#select 4th, 5th and 6th column from dataset and save in y
y = dataset.iloc[:, 3:6].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#custom input for prediction
my_input = np.array([[53.06,87.54,55.85]])

#load the model
model = keras.models.load_model('model/model2000.h5')

predictions = model.predict(my_input)

print(predictions)
q1,q2,q3 = predictions[0][0],predictions[0][1],predictions[0][2]
q1 = float("{:.4f}".format(q1))
q2 = float("{:.4f}".format(q2))
q3 = float("{:.4f}".format(q3))

print(q1,q2,q3)
pred = [[q1,q2,q3]]
# predictions to csv with pandas
df = pd.DataFrame(pred)
#df.to_csv('predictions.csv', index=False, header=False)

