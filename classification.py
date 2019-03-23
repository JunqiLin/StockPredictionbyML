#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:41:32 2019

@author: linjunqi
"""

import tensorflow as tf
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
 
if __name__ == "__main__":
    
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
    n_sma_data = (sma_data - np.mean(sma_data))/np.std(sma_data)
    wma_data = np.array(factors[2])
    n_wma_data = (wma_data - np.mean(wma_data))/np.std(wma_data)
    mom_data = np.array(factors[3])
    lables_data = np.array(factors[5])[:, np.newaxis]
#    features_mat = np.vstack((sma_data,wma_data)).T.astype(np.float32)
    features_mat = np.vstack((n_sma_data,n_wma_data)).T.astype(np.float32)
    print("features matrax datatype is %s its shape is %s"%(type(features_mat),features_mat.shape))
    print("lables data type is %s its shape is %s"%(type(lables_data),lables_data.shape))
    
    
    
    #定义神经网络的参数
    w = tf.Variable(tf.random_normal([2,10],stddev=1,seed=1))
    w1 = tf.Variable(tf.random_normal([10,1],stddev=1,seed=1))
    b = tf.Variable(tf.random_normal([1,10],stddev=1,seed=1))
    #定义输入和输出
    x = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
    y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-input")
    m = tf.matmul(x,w) + b
    yz = tf.nn.softmax(tf.matmul(x,w) + b)
    y = tf.nn.sigmoid(tf.matmul(yz,w1))
    

    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y_) * tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    rdm = RandomState(1)
    dataset_size = 100

    X = features_mat
    Y = lables_data

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        steps = 500
        for i in range(steps):
           for (input_x,input_y) in zip(X,Y):
               input_x = np.reshape(input_x,(1,2))
               input_y = np.reshape(input_y,(1,1))
               sess.run(train_step,feed_dict={x:input_x,y_:input_y})
            #每迭代1000次输出一次日志信息
           if i % 100 == 0:
                # 计算所有数据的交叉熵
                total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
                # 输出交叉熵之和
                print("After %d training step(s),cross entropy on all data is %g"%(i,total_cross_entropy))
        #预测输入X的类别
        pred_Y = sess.run(y,feed_dict={x:X})
        index = 1
        for pred,real in zip(pred_Y,Y):
            print(pred,real)
            
        index = 1
        right = 0
        wrong = 0
        for pred,real in zip(pred_Y,Y):
            if (pred[0] >0.5 and real[0]==1) or (pred[0]<0.5 and real[0]==-1) :
                right += 1
            else :
                wrong += 1
            
        print(pred,real)
        print(right, wrong)
