#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:44:47 2019

@author: linjunqi
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import date
import quandl
import talib
start_date = date(2010,1,3)
end_date = date.today()
apple = quandl.get("WIKI/AAPL", start_date=start_date, end_date=end_date)
train_data=apple[:300]
close_prices = np.array(train_data['Close'])

lables = np.zeros(len(close_prices))
for index,price in enumerate(close_prices):
    if index == 0:
        lables[index] = 0
    else:
        pre_price = close_prices[index-1]
        cur_price = close_prices[index]
        if cur_price>pre_price:
            lables[index] = 1
        else :
            lables[index] = -1
            


date_time = train_data.index
factors = pd.DataFrame(list(zip(date_time,talib.SMA(close_prices),
                                talib.WMA(close_prices),talib.MOM(close_prices),
                                close_prices,lables)))

factors.dropna(axis = 0, how='any',inplace = True)

sma_data = np.array(factors[1])
wma_data = np.array(factors[2])
mom_data = np.array(factors[3])
lables_data = np.array(factors[5])[:, np.newaxis]
features_mat = np.vstack((sma_data,wma_data,mom_data)).T.astype(np.float32)


from numpy.random import RandomState
w = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
b = tf.Variable(tf.random_normal([1],stddev=1,seed=1))
x = tf.placeholder(tf.float32,shape=(None,3),name="x-input")
y = tf.placeholder(tf.float32,shape=(None,1),name="y-input")

y_ = tf.nn.sigmoid(tf.matmul(x,w) + b)
cross_entropy = -tf.reduce_mean(y * tf.log(tf.clip_by_value(y_,1e-10,1.0))+(1-y) * tf.log(tf.clip_by_value(1-y_,1e-10,1.0)))


train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
rdm = RandomState(1)
dataset_size = 100
#XX = rdm.rand(dataset_size,2)
#YY = [[int(x1+x2 < 1)] for (x1,x2) in XX]
X = features_mat
Y = lables_data.tolist()


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    steps = 500
    for i in range(steps):
        for (input_x,input_y) in zip(X,Y):
            input_x = np.reshape(input_x,(1,3))
            input_y = np.reshape(input_y,(1,1))
            
            sess.run(train_step,feed_dict={x:input_x,y:input_y})
        if i % 100 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X ,y:Y})
            print("After %d training step(s),prediction on all data is %s"%(i,sess.run(y_,feed_dict={x: X})))



