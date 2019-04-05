#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:16:15 2019

@author: madanraj
"""
from keras.utils import np_utils
from keras import backend as K
from scipy import io as sio

import tensorflow as tf
import dataset_path
import parameters

tf.reset_default_graph()

print('backend', K.backend())


#[Dataset taken from : https://www.nist.gov/itl/iad/image-group/emnist-dataset]

'''
Load dataset in matlab format
'''
def load_input(input_index):
    mat = sio.loadmat(dataset_path.data_path[input_index])
    data = mat['dataset']
    return data

'''
Extract training and test images and labels
'''
def extract_train_test(data):
    X_train = data['train'][0, 0]['images'][0, 0]
    y_train = data['train'][0, 0]['labels'][0, 0]
    X_test = data['test'][0, 0]['images'][0, 0]
    y_test = data['test'][0, 0]['labels'][0, 0]
    return X_train, y_train, X_test, y_test

'''
Reshape data to match the number of rows and cols in images
'''
def reshape_data(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], 28, 28, order = "A")
    X_test = X_test.reshape(X_test.shape[0], 28, 28, order = "A")
    return X_train, X_test

'''
Convert data to binary format matrix
'''
def convert_data(X_train, X_test, y_train, y_test):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, parameters.no_classes)
    Y_test = np_utils.to_categorical(y_test, parameters.no_classes)
    return X_train, X_test, Y_train, Y_test

'''
Print utility to print the data stats
'''
def print_data_stats(X_train, y_train, X_test, Y_train):
    print()
    print('MNIST data loaded: train:', len(X_train), 'test:', len(X_test))
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('Y_train:', Y_train.shape)
    
def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)
    outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
    
    
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']



if __name__ == '__main__':

    data = load_input(0)
    X_train, y_train, X_test, y_test = extract_train_test(data)
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    X_train, X_test = reshape_data(X_train, X_test)
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    X_train, X_test, Y_train, Y_test = convert_data(X_train, X_test, y_train, y_test)
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    
    # Training Parameters
    learning_rate = 0.001
    training_steps = 10000
    batch_size = 128
    display_step = 200
    
    # Network Parameters
    num_input = 28 # MNIST data input (img shape: 28*28)
    timesteps = 28 # timesteps
    num_hidden = 128 # hidden layer num of features
    num_classes = 10 # MNIST total classes (0-9 digits)
    
    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    
    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    
    
    
    logits = RNN(X, weights, biases)
    prediction = tf.nn.softmax(logits)
    
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    
    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    epochs = 2
    
    with tf.Session() as sess:

        sess.run(init)
        
        while epochs > 0:
            # Run the initializer
            start = 0
            end = batch_size
            for step in range(1, 467):
                batch_x = X_train[start:end]
                batch_y = Y_train[start:end]
                
                start = end
                end = end + batch_size
                # Reshape data to get 28 seq of 28 elements
                batch_x = batch_x.reshape((batch_size, timesteps, num_input))
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
               
                
                if step%10 == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
                    
                
        
            print("Optimization Finished!")
            
            epochs = epochs - 1
    
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: X_test, Y: Y_test}))