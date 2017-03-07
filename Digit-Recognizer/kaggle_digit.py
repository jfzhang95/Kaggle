#!usr/bin/env python
#-*- coding:utf-8 -*-
#__author__ = 'Jesse Zhang'


import csv
import numpy as np
import operator

def Arr2Int(array):
    array = np.mat(array)
    m, n = np.shape(array)
    newArray = np.zeros((m, n))
    for i in xrange(m):
        for j in xrange(n):
            newArray[i, j] = int(array[i, j])
    return newArray


def binaryzation(array):
    m, n = np.shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i, j] != 0:
                array[i, j] = 1
    return array

def LoadTrainData():

    all = []
    with open('train.csv') as file:
        text = csv.reader(file)
        for line in text:
            all.append(line)   # 42001*785
    all.remove(all[0])    # remove first row
    all = np.array(all)
    labels = all[:, 0]   # 1*42000
    data = all[:, 1:]   # 42000*784

    traindata = binaryzation(Arr2Int(data))
    trainlabels = Arr2Int(labels)
    print 'training data is ready!'
    return traindata, trainlabels



def LoadTestData():

    all = []
    with open('test.csv') as file:
        text = csv.reader(file)
        for line in text:
            all.append(line)   # 42001*785
    all.remove(all[0])    # remove first row
    all = np.array(all)
    labels = all[:, 0]   # 1*42000
    data = all[:, 0:]   # 42000*784

    testdata = binaryzation(Arr2Int(data))
    print 'testing data is ready!'
    return testdata

def SaveResult(result):
    with open('submission.csv', 'wb') as MyFile:
        myWriter = csv.writer(MyFile)
        myWriter. writerow(["ImageId", "Label"])
        index=0
        for i in result:
            tmp = []
            index = index + 1
            tmp.append(index)
            tmp.append(int(i))
            myWriter.writerow(tmp)


def Knn_Classify(traindata, trainlabels, testdata, k):

    datasize = traindata.shape[0]
    SqDistances = 0.0
    distances = []
    for i in xrange(datasize):
        SqDistances = np.sum((traindata[i] - testdata) ** 2)
        Distances = np.sqrt(SqDistances)
        distances.append(Distances)
    distances = np.array(distances)
    sortedDistIndicies = distances.argsort()
    classcount = {}
    for i in range(k):
        votelabel = trainlabels[0, sortedDistIndicies[i]]
        classcount[votelabel] = classcount.get(votelabel, 0) + 1
    sortedClassCount = sorted(classcount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

def run():
    traindata, trainlabels = LoadTrainData()
    testdata = LoadTestData()
    m = testdata.shape[0]
    testlabels = []
    for i in xrange(m):
        label = Knn_Classify(traindata, trainlabels, testdata[i], 5)
        testlabels.append(label)
        if i % 100 == 0:
            print 'continue', i, '.....'
    SaveResult(testlabels)
    print "finish!"


run()