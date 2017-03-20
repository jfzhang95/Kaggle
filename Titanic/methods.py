#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date:   
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from logRegres import *
from LoadData import X_train, X_test, Y_train, test
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score
import csv


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
