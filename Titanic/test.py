#!usr/bin/env python
#-*- coding:utf-8 -*-

"""
@author: James Zhang
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from logRegres import *
from LoadData import X_train, X_test, Y_train

def logregression(X_train, X_test, Y_train):
    X = []
    Y = []

    for ele in X_train:
        X.append(ele)
    for ele in Y_train:
        Y.append(ele)
    X = np.array(X)
    Y =  np.array(Y)
    # w = logregres_train(X, Y)
    # print w

    total_acc = 0.0
    iters = 10
    for i in xrange(iters):
        weights = logregres_train(X, Y)
        acc = logregres_test(X, Y, weights)
        total_acc += np.float(acc)

    print "mean_acc:", total_acc / iters, "%"

    X_test = np.array(X_test)

    predict = logregres_predict(X_test, weights)

    print predict

# run
logregression(X_train, X_test, Y_train)
