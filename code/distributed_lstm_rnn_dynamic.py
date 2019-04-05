#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 02:09:54 2019

@author: madanraj
"""

import tensorflow as tf
import time
from scipy import io as sio
from keras.utils import np_utils

import dataset_path
import parameters

tf.reset_default_graph()

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)

# cluster specification
parameter_servers = ["localhost:8117"]
workers = ["localhost:8124","localhost:8125"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name","worker", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index",1, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index)

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
    
def RNN(x):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)    
    
    lstm_cell = tf.contrib.rnn.LSTMBlockCell(num_hidden, forget_bias=1.0)
    
    outputs, states = tf.nn.dynamic_rnn(
        cell=lstm_cell, inputs=x, time_major=False, dtype=tf.float32)
    
    output_layer = tf.layers.Dense(
        num_classes, activation=None, 
        kernel_initializer=tf.orthogonal_initializer()
    )
    return output_layer(tf.layers.batch_normalization(outputs[:, -1, :]))


if FLAGS.job_name == "ps":
    print("server")
    server.join()

elif FLAGS.job_name == "worker":
    
    print("worker")
    
    #Training Parameters
    learning_rate = 0.001
    training_steps = 10000
    batch_size = 128
    display_step = 200
    
    # Network Parameters
    num_input = 28 # MNIST data input (img shape: 28*28)
    timesteps = 28 # timesteps
    num_hidden = 128 # hidden layer num of features
    num_classes = 10 # MNIST total classes (0-9 digits)
    
    frequency = 100
    batch_size = 100
    learning_rate = 0.0005
    training_epochs = 2
    
    data = load_input(0)
    X_train, y_train, X_test, y_test = extract_train_test(data)
    X_train, X_test = reshape_data(X_train, X_test)
    X_train, X_test, Y_train, Y_test = convert_data(X_train, X_test, y_train, y_test)
    

    with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d" % FLAGS.task_index,
		cluster=cluster)):
        
        global_step = tf.get_variable(
               'global_step',
            [],
            initializer = tf.constant_initializer(0),
		   trainable = False)
        
        
        # tf Graph input
        X = tf.placeholder("float", [None, timesteps, num_input])
        Y = tf.placeholder("float", [None, num_classes])
    
        logits = RNN(X)
        prediction = tf.nn.softmax(logits)
    
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=Y))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        gvs = optimizer.compute_gradients(loss_op)
        capped_gvs = [
                (tf.clip_by_norm(grad, 2.), var) if not var.name.startswith("dense") else (grad, var)
                for grad, var in gvs]
        for _, var in gvs:
            if var.name.startswith("dense"):
                print(var.name)    
        train_op = optimizer.apply_gradients(capped_gvs)
    
        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

	
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
														global_step=global_step,
														init_op=init_op)
    
   
    
    with sv.prepare_or_wait_for_session(server.target) as sess:
      
        # perform training cycles
        start_time = time.time()
        while training_epochs > 0:
            print("running epochs")
            start = 0
            end = batch_size
            count = 0
            for step in range(1, 467):
                batch_x = X_train[start:end]
                batch_y = Y_train[start:end]
                start = end
                end = end + batch_size
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
               
                
                if step%10 == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
                    
                
        
            print("Optimization Finished!")
                    
            training_epochs = training_epochs - 1
            
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: X_test, Y: Y_test}))
         
    sv.stop()
    print("done")