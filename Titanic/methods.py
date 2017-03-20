#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date:   2017-03-20
"""

import numpy as np
from logRegres import *
from LoadData import X_train, X_test, Y_train, test
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score
import csv
import operator

# logistic regression
def logregression(X_train, X_test, Y_train):
    X = []
    Y = []

    for ele in X_train:
        X.append(ele)
    for ele in Y_train:
        Y.append(ele)
    X = np.array(X)
    Y =  np.array(Y)
    total_acc = 0.0
    iters = 10
    for i in xrange(iters):
        weights = logregres_train(X, Y)
        acc = logregres_test(X, Y, weights)
        total_acc += np.float(acc)

    print "mean_acc:", total_acc / iters, "%"
    X_test = np.array(X_test)
    Y_pred = logregres_predict(X_test, weights)
    # write result
    with open('logreg_submission.csv', 'wb') as MyFile:
        myWriter = csv.writer(MyFile)
        myWriter.writerow(["PassengerId", "Survived"])
        index = 0
        for i in Y_pred:
            tmp = []
            tmp.append(test['PassengerId'][index])
            tmp.append(int(i))
            myWriter.writerow(tmp)
            index += 1

def RandomForest(X_train, Y_train, X_test):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, Y_train)
    print cross_val_score(rfc, X_train, Y_train, cv=5).mean()
    Y_pred = rfc.predict(X_test)
    # write result
    with open('randomforest_submission.csv', 'wb') as MyFile:
        myWriter = csv.writer(MyFile)
        myWriter.writerow(["PassengerId", "Survived"])
        index = 0
        for i in Y_pred:
            tmp = []
            tmp.append(test['PassengerId'][index])
            tmp.append(int(i))
            myWriter.writerow(tmp)
            index += 1

def XGBoosting(X_train, Y_train, X_test):
    xgbc = XGBClassifier(n_estimators=100, learning_rate=1e-1, max_depth=5)
    cross_val_score(xgbc, X_train, Y_train, cv=5).mean()
    xgbc.fit(X_train, Y_train)
    Y_pred = xgbc.predict(X_test)
    with open('xgboosting_submission.csv', 'wb') as MyFile:
        myWriter = csv.writer(MyFile)
        myWriter.writerow(["PassengerId", "Survived"])
        index = 0
        for i in Y_pred:
            tmp = []
            tmp.append(test['PassengerId'][index])
            tmp.append(int(i))
            myWriter.writerow(tmp)
            index += 1

def KNN(X_train, Y_train, X_test):
    # 归一化
    def autoNorm(dataSet):
        dataSet = np.array(dataSet)
        min_value = dataSet.min(0)  # 取出数据集中的最小值
        max_value = dataSet.max(0)  # 取出数据集中的最大值
        range = max_value - min_value  # 计算取值范围
        normDataSet = np.zeros(np.shape(dataSet))  # 初始化一个矩阵，该矩阵和所给数据集维度相同用于存放归一化之后的数据
        m = dataSet.shape[0]  # 取出数据集的行数
        normDataSet = dataSet - np.tile(min_value, (m, 1))  # 这里tile()函数创建了一个以min_value为值的m行列向量，然后计算oldValue-min_value
        normDataSet = normDataSet / np.tile(range, (m, 1))  # 特征值相除得到归一化后的数值
        return normDataSet  # 返回归一化后的数据



    def KnnClassifier(X_train, Y_train, x_test, k):
        datasize = X_train.shape[0]
        distances = []
        for i in xrange(datasize):
            SqDistances = np.sum((X_train[i] - x_test) ** 2)
            Distances = np.sqrt(SqDistances)
            distances.append(Distances)
        distances = np.array(distances)
        sortedDistIndicies = distances.argsort()
        classcount = {}
        for i in range(k):
            votelabel = Y_train[sortedDistIndicies[i]]
            classcount[votelabel] = classcount.get(votelabel, 0) + 1
        sortedClassCount = sorted(classcount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def SaveResult(result):
        with open('Knn_submission.csv', 'wb') as MyFile:
            myWriter = csv.writer(MyFile)
            myWriter.writerow(["PassengerId", "Survived"])
            index = 0
            for i in result:
                tmp = []
                tmp.append(test['PassengerId'][index])
                tmp.append(int(i))
                myWriter.writerow(tmp)
                index += 1

    X_train = autoNorm(X_train)
    X_test = autoNorm(X_test)
    Y_train = np.array(Y_train)

    test_size = X_test.shape[0]
    testlabels = []
    for i in xrange(test_size):
        label = KnnClassifier(X_train, Y_train, X_test[i], 30)
        testlabels.append(label)
        if i % 100 == 0:
            print 'continue', i, '.....'
    SaveResult(testlabels)
    print "finish!"
