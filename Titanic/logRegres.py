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
        # h = sigmoid(data * weights)
        h = sigmoid(np.dot(data, weights))
        labels = labels.reshape(891, 1)
        error = labels - h
        # print np.shape(error)
        # print np.shape(data.transpose())
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
    # print "accuracy:", accuracy * 100, "%"
    # print 'done!'
    return accuracy

def logregres_predict(data, weights):
    m, n = np.shape(data)
    predict = []
    for i in xrange(m):
        pred = sigmoid(np.dot(data[i, :], weights)) > 0.5
        # print pred
        if pred[0] == True:
            predict.append(1)
        else:
            predict.append(0)
    return predict

def showLogRegres(weights, train_x, train_y):
    # notice: train_x and train_y is mat datatype
    numSamples, numFeatures = np.shape(train_x)
    train_y = train_y.transpose()
    if numFeatures != 3:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

        # draw all samples
    for i in xrange(numSamples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

            # draw the classify line
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]

    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


