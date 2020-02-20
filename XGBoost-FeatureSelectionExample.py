import os,re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer # Process NaN in data
from sklearn.utils import shuffle

import numpy as np

import xgboost as xgb
import operator

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import warnings
warnings.filterwarnings('ignore')

# In[]:
#-----------------------------------------------------------------------------# 
def Individual_mean_std_normalization(x_train, x_test):
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_test_mean = x_test.mean(axis=0)
    x_test_std = x_test.std(axis=0)
    x_train_Norm = (x_train - x_train_mean) / x_train_std
    x_test_Norm = (x_test - x_test_mean) / x_test_std
    return x_train_Norm, x_test_Norm
def Individual_mean_std_normalization2(x_train):
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train_Norm = (x_train - x_train_mean) / x_train_std
    return x_train_Norm

# In[]:
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
# In[]: xgboost screen

train = df0 # Training Data, data type shold be ndarray
test = df1 # Validation Data
test1 = df2

params = {
        'min_child_weight': 100,
        'eta': 0.02,
        'colsample_bytree': 0.7,
        'max_depth': 12,
        'subsample': 0.7,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'verbose_eval': True,
        'seed': 12}
rounds = 10

y= train['label']
X = train .drop(['ID','label'],axis=1)

xgtrain = xgb.DMatrix(X, label=y)
bst = xgb.train(params, xgtrain, num_boost_round=rounds)

features = [x for x in train.columns if x not in ['ID','label']]
ceate_feature_map(features)

importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
df.to_csv('./xgboostSelect/feat_importance.csv', index=False)

plt.figure()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.show()

