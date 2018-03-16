# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:04:56 2018

Tensorflow for RT prediction
Adapted by r1.1 get started

@author: W Wei
"""

import tensorflow as tf
import pandas as pd

trainData = "../data/train-data.csv"
testData = "../data/test-data.csv"
trainData2 = "../data/train-data2.csv"

def load_data2(filename):
    train = pd.read_csv(filename, header=0) 
    MeasredTmf = train.pop("Measured-Tmf")
    vovera = train
    
    return (vovera, MeasredTmf)
    
def main():   
    
    # Model parameters
    W = tf.Variable([0.05], tf.float32)
    b = tf.Variable([0.4], tf.float32)
    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    # training data
    (vovera, MeasredTmf) = load_data2(trainData2)
    print(vovera)
    print(MeasredTmf)
    x_train = vovera
    y_train = MeasredTmf
    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong
    for i in range(10):
      sess.run(train, {x:x_train, y:y_train})
    
    # evaluate training accuracy
    curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    
#    print(sess.run(linear_model, {x:vovera}))


if __name__=="__main__":
    main()