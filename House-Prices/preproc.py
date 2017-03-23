#preprocessing for training&test data
#@2016.11.08

import pandas as pd
#step1:reading csv data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#train.head()   # take a brief look at training data
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))  # concat training&test data

import numpy as np
from scipy.stats import skew
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#step2:log transform for training data (including the labels)
'''  a png for labels' distribution
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()
plt.savefig('label_dist.png',dpi=150)
'''
train["SalePrice"] = np.log1p(train["SalePrice"]) #log transform the target

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index   # get the index of all the numeric features
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]   # get the features whose skewness>0.75
skewed_feats = skewed_feats.index # get the index of the features whose skewness>0.75
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])  #log transform for all those features

#step3: transform for the categorial features
all_data = pd.get_dummies(all_data)

#step4: fill all the missing value
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

# save the handled data to file
X_train.to_csv('train_features',index=False) #index=False means not store the index of each line
X_test.to_csv('test_features',index=False)
y.to_csv('train_labels',index=False)
# attention: if u donnt want to store the header,use header=False,
# and then when u read it u need to use read_csv('xx.csv',header=None)
