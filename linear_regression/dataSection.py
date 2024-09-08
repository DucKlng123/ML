import matplotlib
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')

X,y = mnist['data'],mnist['target']
print('X.shape',X.shape)
print('y.shape',y.shape)

#切分训练集和测试集
X_train , X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#洗牌
shuffle_index = np.random.permutation(60000)
X_train , y_train = X_train.loc[shuffle_index], y_train.loc[shuffle_index]
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier( max_iter= 5,random_state= 42)
sgd_clf.fit(X_train,y_train_5)

# print ('predict',sgd_clf.predict([X.loc[35000]]))
# print ('real state',y.loc[35000] == '5')
#
# from sklearn.model_selection import cross_val_score
# accuracy = cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring='accuracy')
# print (accuracy)



