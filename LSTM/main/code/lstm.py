#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:31:56 2019

@author: linjunqi
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import os 
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
import matplotlib.pyplot as plt
tf.reset_default_graph()
D = 1 
num_unrollings = 20
 
batch_size = 200
num_nodes = [20,20,15] 
n_layers = len(num_nodes) 
dropout = 0.2 

mainDirectory = str("./model_1/")

#trainFiles = [f for f in listdir("./train/") if isfile(join("./train/", f))]
#evalFiles = [f for f in listdir("./eval/") if isfile(join("./eval/", f))]

scaler = preprocessing.MinMaxScaler()

#for fileName in trainFiles:
#    trainDataWhole = pd.read_csv("./train/" + fileName, sep=',')
#    trainDataClose = np.array(trainDataWhole['Close'])[:,np.newaxis]
#    trainData = scaler.fit_transform(trainDataClose).reshape(-1,1)
#    
#for fileName in evalFiles:
#    evalDataWhole = pd.read_csv("./eval/" + fileName, sep=',')
#    evalDataClose = np.array(evalDataWhole['Close'])[:, np.newaxis]
#    evalData = scaler.fit_transform(evalDataClose).reshape(-1,1)
df=pd.read_csv(os.getcwd()+'/train/dataset_1.csv')
trainData=np.array(df['close'])
trainData=(trainData-np.mean(trainData))/np.std(trainData)
l = len(trainData)
trainData = trainData.reshape(l,1)



train_inputs_w, train_outputs_w = [],[]


for i in range(len(trainData)-num_unrollings-1):
    train_inputs_w.append(np.array(trainData[i:i+num_unrollings]))
    train_outputs_w.append(np.array(trainData[i+1:i+num_unrollings+1]))

def generate_batch(time):
    iter_ = time*batch_size
    x_batch = np.array(train_inputs_w[iter_:iter_+batch_size]).astype(np.float32)
    y_batch = np.array(train_outputs_w[iter_:iter_+batch_size]).astype(np.float32)
    
    return x_batch,y_batch


#
train_inputs = tf.placeholder(tf.float32, shape=[batch_size,num_unrollings, D])
train_outputs = tf.placeholder(tf.float32, shape=[batch_size,num_unrollings,D])

lstm_cells = [
   tf.contrib.rnn.LSTMCell(num_units=num_nodes[li],
                           state_is_tuple=True,
                           initializer= tf.contrib.layers.xavier_initializer()
                           #initialize the weight
                          )
for li in range(n_layers)]


drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
   lstm, input_keep_prob=1.0,output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout
) for lstm in lstm_cells]

drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)


w = tf.get_variable('w',shape=[num_nodes[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable('b',initializer=tf.random_uniform([1],-0.1,0.1))


#

c, h = [],[]
initial_state = []
for li in range(n_layers):
    c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    initial_state.append(tf.contrib.rnn.LSTMStateTuple(c[li], h[li]))



#all_inputs = tf.concat([tf.expand_dims(t,0) for t in train_inputs],axis=0)
all_inputs = train_inputs

all_lstm_outputs, state = tf.nn.dynamic_rnn(
   drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
   time_major = False, dtype=tf.float32)
#batchsize*max_time*...
all_lstm_outputs = tf.reshape(all_lstm_outputs, [-1,num_nodes[-1]])

all_outputs = tf.nn.xw_plus_b(all_lstm_outputs,w,b)



print('Defining training Loss')
with tf.name_scope('accuracy'):
    loss = 0.0
#with tf.control_dependencies([tf.assign(c[li], state[li][0]) for li in range(n_layers)]+
#                            [tf.assign(h[li], state[li][1]) for li in range(n_layers)]):
    loss=tf.reduce_mean(tf.square(tf.reshape(all_outputs,[-1])-tf.reshape(train_outputs, [-1])))
    tf.summary.scalar('loss', loss)
    

# Optimizer.
print('TF Optimization operations')
optimizer = tf.train.AdamOptimizer(0.001)
gradients, v = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
optimizer = optimizer.apply_gradients(zip(gradients, v))



with tf.Session() as sess:
    merged = tf.summary.merge_all()
    trainWriter = tf.summary.FileWriter(
                os.getcwd()+'/train', sess.graph, flush_secs=1, max_queue=2)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    results =[]
    for _ in range(100):
        for i in range(25):
            x,y = generate_batch(i)
            summary,all_outputs_,loss_ = sess.run([merged,all_outputs,loss],feed_dict={all_inputs:x, train_outputs:y})
            results.append(loss_)
            if i == 1:
                print(loss_) 
            trainWriter.add_summary(summary, i)
            trainWriter.flush()
    





#print('Defining prediction related TF functions')
#
#sample_inputs = tf.placeholder(tf.float32, shape=[1,D])
#
#sample_c, sample_h, initial_sample_state = [],[],[]
#for li in range(n_layers):
# sample_c.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
# sample_h.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
# initial_sample_state.append(tf.contrib.rnn.LSTMStateTuple(sample_c[li],sample_h[li]))
#
#reset_sample_states = tf.group([tf.assign(sample_c[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)],
#                              [tf.assign(sample_h[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)])
#
#sample_outputs, sample_state = tf.nn.dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs,0),
#                                  initial_state=tuple(initial_sample_state),
#                                  time_major = True,
#                                  dtype=tf.float32)
#
#with tf.control_dependencies([tf.assign(sample_c[li],sample_state[li][0]) for li in range(n_layers)]+
#                             [tf.assign(sample_h[li],sample_state[li][1]) for li in range(n_layers)]):  
# sample_prediction = tf.nn.xw_plus_b(tf.reshape(sample_outputs,[1,-1]), w, b)
#
#print('\tAll done')
#
#epochs = 30
#valid_summary = 1 
#n_predict_once = 50
#train_seq_length = trainData.size
#train_mse_ot = [] 
#test_mse_ot = [] 
#predictions_over_time = []
#session = tf.InteractiveSession()
#tf.global_variables_initializer().run()
#loss_nondecrease_count = 0
#loss_nondecrease_threshold = 2 
#
#print('Initialized')
#average_loss = 0
#data_gen = DataGeneratorSeq(train_data,batch_size,num_unrollings) 
#x_axis_seq = []
#test_points_seq = np.arange(11000,12000,50).tolist() 
#
#for ep in range(epochs):       
#   
#   # ========================= Training =====================================
#   for step in range(train_seq_length//batch_size):
#       
#       u_data, u_labels = data_gen.unroll_batches()
#
#       feed_dict = {}
#       for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):            
#           feed_dict[train_inputs[ui]] = dat.reshape(-1,1)
#           feed_dict[train_outputs[ui]] = lbl.reshape(-1,1)
#       
#       feed_dict.update({tf_learning_rate: 0.0001, tf_min_learning_rate:0.000001})
#
#       _, l = session.run([optimizer, loss], feed_dict=feed_dict)
#
#    




  
