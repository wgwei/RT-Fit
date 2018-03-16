# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 09:53:17 2018
test pandas read
@author: weigang
"""

import pandas as pd
import tensorflow as tf

trainData = "../data/train-data.csv"
testData = "../data/test-data.csv"
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
    
    
    # Feature Columns
    print (train_x)
    print(train_y)
    training_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_x, y=train_y, batch_size=1000, shuffle=True, num_epochs=None)
    
    eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_x, y=test_y, batch_size=1, shuffle=False, num_epochs=None)
    
    print(test_x)
    print(test_y)

    dnn_features = my_feature_columns
    print(dnn_features)
    # regressor = tf.contrib.learn.LinearRegressor(feature_columns=[trans])
    
    dnnregressor = tf.contrib.learn.DNNRegressor(feature_columns=dnn_features, hidden_units=[16, 8])
    
    #train the model
    dnnregressor.fit(input_fn=training_input_fn, steps=1)
    
    # Evaluate the trianing
    dnnregressor.evaluate(input_fn=eval_input_fn, steps=1)    
    
    # Predictions
    predictdf = pd.read_csv('../data/predict-data.csv', names=colms[0:-1], header=0)
    predict_input_fn = tf.estimator.inputs.pandas_input_fn(x=predictdf,shuffle=False, num_epochs=1)
    
    print("Predicting scores **********************")
    
    
    y = dnnregressor.predict_scores(input_fn=predict_input_fn)
    for x in y:
        print(x)
        print('\n')
    
    
if __name__=="__main__":
    main()

     
     