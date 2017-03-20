#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
"""


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))

def logregres_train(data, labels):
    labels = np.array(labels)
    labels = labels.transpose()
    m, n = np.shape(data)
    alpha = 1e-3
    iters = 5000
    weights = np.random.randn(n, 1)
    for iter in xrange(iters):
        h = sigmoid(np.dot(data, weights))
        labels = labels.reshape(891, 1)
        error = labels - h
        weights += alpha * np.dot(data.transpose(), error)

    return weights

def logregres_test(data, labels, weights):
    labels = labels.transpose()
    m, n = np.shape(data)
    acc_count = 0
    for i in xrange(m):
        predict = sigmoid(np.dot(data[i, :], weights)) > 0.5
        if predict.tolist()[0] == bool(labels[i]):
            acc_count += 1
    accuracy = float(acc_count) / m
    return accuracy

def logregres_predict(data, weights):
    m, n = np.shape(data)
    predict = []
    for i in xrange(m):
        pred = sigmoid(np.dot(data[i, :], weights)) > 0.5
        if pred[0] == True:
            predict.append(1)
        else:
            predict.append(0)
    return predict



