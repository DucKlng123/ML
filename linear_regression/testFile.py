import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import pivot
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def setDatasets(flg_x,flg_y):
    global X,y,X_b
    #x
    if flg_x == 'uniVar':
        X = 6 * np.random.rand(100,1) -3
        theta = [[4],[3]]
    elif flg_x == 'doubleVar':
        X = 3 * np.random.rand(100, 2)
        theta = [[4],[3],[2]]
    X_b = np.c_[np.ones((100, 1)), X]

    #y
    if flg_y == 'flg_linear' :
        y =  X_b.dot(theta) + np.random.randn(100,1)
    elif flg_y == 'flg_quadratic':
        y = (X_b**2).dot(theta) + np.random.randn(100,1)

setDatasets('uniVar','flg_quadratic')

plt.scatter(X,y)
plt.xlabel('x')
plt.ylabel('y')
def train(learning_rate=0.1,num_iters=1000):
    np.random.seed(42)
    theta = np.random.randn(2,1)
    m = len(y)
    for i in range(num_iters):
        gradient = 2/m * X_b.T.dot(X_b.dot(theta)-y)
        theta = theta - learning_rate * gradient
    return theta,(X_b.dot(theta)-y).T.dot((X_b.dot(theta)-y))/m


theta,cost = train()
x_test = np.linspace(X.min(),X.max(),100).reshape(100,1)
x_test_b = np.c_[np.ones((100,1)),x_test]
y_predict = x_test_b.dot(theta)
plt.scatter(x_test, y_predict)
plt.show()
print ('theta 1',theta)
print ('cost 1',cost[0][0])


#minibatch gradient descent
theta_path_mgd = []
n_iterations = 50
minibatch = 16

def learning_schedule(t):
    return 10/(50+t)

def train_mgd(n_iterations):
    t = 0
    m = len(y)
    theta = np.random.randn(2, 1)
    for epoch in range(n_iterations):
        shuffle_index = np.random.permutation(len(y))
        X_shuffled = X_b[shuffle_index]
        y_shuffled = y[shuffle_index]
        for i in range(0,len(y),minibatch):
            t += 1
            xi = X_shuffled[i:i+minibatch]
            yi = y_shuffled[i:i+minibatch]
            gradient = 2/minibatch * xi.T.dot(xi.dot(theta)-yi)
            theta = theta - learning_schedule(t) * gradient
            theta_path_mgd.append(theta)

    return theta , ( X_b.dot(theta)-y ).T.dot( X_b.dot(theta)-y )/m

theta,cost = train_mgd(1000)
y_predict = x_test_b.dot(theta)
plt.scatter(x_test, y_predict)
plt.scatter(X,y)
plt.axis([-3,3,3,16])
plt.show()
print ('theta 2',theta)
print ('cost 2',cost[0][0])



# 多项式
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(poly_features.get_feature_names_out(['x1']))
print(lin_reg.coef_)
print(lin_reg.intercept_)

X_test = np.linspace(-3,3,100).reshape(100,1)
X_test_poly = poly_features.transform(X_test)
y_test = lin_reg.predict(X_test_poly)
plt.plot(X,y,'b.',label = 'training set')
plt.plot(X_test,y_test,'--',label = 'prediction')
plt.legend()
plt.axis([-3,3,3,16])
plt.show()

def contrast_plot():
    plt.figure(figsize=(10,5))
    for style,width,degree in [('g--',1,1),('b-+',1,2),('r-.',1,100)] :
        poly_features = PolynomialFeatures(degree=degree)
        std = StandardScaler()
        lin_reg = LinearRegression()
        polynomial_reg = Pipeline([('poly_features',poly_features),
                                   ('std_scaler',std),
                                   ('lin_reg',lin_reg)])
        polynomial_reg.fit(X,y)
        y_poly_pred = polynomial_reg.predict(X_test)
        plt.plot(X_test,y_poly_pred,style,label = f'degree {degree}')

        #get mean square error


    plt.plot(X,y,'b.')
    plt.legend()
    plt.axis([-3,3,3,16])
    plt.show()


