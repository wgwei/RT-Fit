# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 09:53:17 2018
test pandas read
@author: weigang
"""

import pandas as pd
import tensorflow as tf

trainData = "train-data.csv"
testData = "test-data.csv"
labelKey = "Measured-Tmf"
colms = ["Volume", "Floor-AbsArea", "Wall-AbsArea", "Ceiling-AbsArea", "Measured-Tmf"]


def load_data(trainFile, testFile):
    # names=colms, read data by colms names,
    # header=0, skip the first row
    # train.pop(columName) read the column by the column name and remove this column from the original data set
    train = pd.read_csv(trainData, names=colms, header=0) 
    train_features, train_label = train, train.pop(labelKey)
    
    test = pd.read_csv(testData, names=colms, header=0)
    test_features, test_label = test, test.pop(labelKey)
     
    return (train_features, train_label), (test_features, test_label)
    
def main():
    (train_x, train_y), (test_x, test_y) = load_data(trainData, testData)
    my_feature_columns = []
    for key in train_x.keys():
        print(key)
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
    classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=3)



from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import itertools
import pandas as pd
import numpy as np
import tensorflow as tf

print('Running version of tensorflow')
print(tf.__version__)

tf.logging.set_verbosity(tf.logging.DEBUG)

names = [
    'trans',
    'price',
]

predict_names = [
    'trans'
]

dtypes = {
    'trans': str,
    'price': np.float32,
}

df = pd.read_csv('simple.csv', names=names, dtype=dtypes, na_values='?')

# Split the data into a training set and an eval set.
training_data = df[:50]
eval_data = df[50:]
print("Training with this :\n")
print(training_data)

# Separate input features from labels
training_label = training_data.pop('price')
eval_label = eval_data.pop('price')

# Feature Columns
training_input_fn = tf.estimator.inputs.pandas_input_fn(x=training_data, y=training_label, batch_size=1, shuffle=True, num_epochs=None)

eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=eval_data, y=eval_label, batch_size=1, shuffle=False, num_epochs=None)

#Embed the column since its a string
transformed_trans = tf.feature_column.categorical_column_with_hash_bucket('trans', 50)
print("Transformed words **********************")
print(transformed_trans)

dnn_features = [tf.feature_column.indicator_column(transformed_trans)]
# regressor = tf.contrib.learn.LinearRegressor(feature_columns=[trans])

dnnregressor = tf.contrib.learn.DNNRegressor(feature_columns=dnn_features, hidden_units=[50, 30, 10])

#train the model
dnnregressor.fit(input_fn=training_input_fn, steps=1)

# Evaluate the trianing
dnnregressor.evaluate(input_fn=eval_input_fn, steps=1)




# Predictions
predictdf = pd.read_csv('simple_predict.csv', names=names, dtype=dtypes, na_values='?')
predict_input_fn = tf.estimator.inputs.pandas_input_fn(x=predictdf,shuffle=False, num_epochs=1)

print("Predicting scores **********************")


y = dnnregressor.predict_scores(input_fn=predict_input_fn)
for x in y:
    print(str(x)+"\n")
    
if __name__=="__main__":
    main()

     
     