from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools 
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO) #gotta git all that infos
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"] #this is all the data and answers
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"] #This is all the features
LABEL = "medv" #this is what should come out

training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,skiprows=1, names=COLUMNS)#parses the CSV
test_set = pd.read_csv("boston_test.csv", skipinitialspace = True, skiprows = 1, names = COLUMNS)#the initial row is the identifier, so remove that
prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace = True, skiprows = 1, names = COLUMNS)

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES] #this makes a special column for each feature, thats why its iterated, used for training

regressor = tf.estimator.DNNRegressor(feature_columns = feature_cols, hidden_units=[10,10],model_dir = "tmp/bostonModel") #this makes the regressor instance, makes two hidden layers, and feature columns that has all the features



def get_input_fn(data_set, num_epochs=None,shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y = pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle) 

y = regressor.predict(
    input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
predictions = list(p["predictions"] for p in itertools.islice(y,6))
print("Predictions: {0}".format(str(predictions)))

