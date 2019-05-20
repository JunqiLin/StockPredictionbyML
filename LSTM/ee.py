#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:04:54 2019

@author: linjunqi
"""
import numpy as np
xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
seq = np.sin(xs)[:, :, np.newaxis]
l_in_x = seq.reshape([-1, 1])
w = np.arange(10).reshape(1,10)
l_in_y = np.dot(l_in_x, w)

l_in_y = l_in_y.reshape([-1, 20, 10])
print(l_in_y.shape)