import numpy as np
from scipy.optimize import minimize

from utils.features import  prepare_for_training
from utils.hypothesis import sigmoid

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

class LogisticRegression:
    def __init__(self, data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data = True):
        self.data,self.feature_mean,self.feature_deviation = prepare_for_training(data,polynomial_degree,sinusoid_degree,normalize_data)
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.theta = np.zeros((self.data.shape[1],self.unique_labels.shape[0]))
        self.theta = self.theta.reshape(self.data.shape[1],self.unique_labels.shape[0])
        self.m = self.data.shape[0]
        self.n = self.data.shape[1]
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

    def train(self,learning_rate = 0.01,iterations = 1000):
        cost_history = np.zeros((iterations,self.unique_labels.shape[0]))
        for (label_index, unique_label) in enumerate(self.unique_labels):
            theta_i = self.theta[:,label_index]
            label_i_bin = (self.labels == unique_label).astype(float)
            (self.theta[:,label_index],cost_history[:,label_index]) = self.gradient_descent(theta_i,label_i_bin,learning_rate,iterations)
        return cost_history

    def gradient_descent(self,theta_i,label_i_bin,learning_rate,iterations):
        cost_history = []
        for i in range(iterations):
            hypothesis = sigmoid(np.dot(self.data,theta_i))
            error = hypothesis - label_i_bin
            gradient = np.dot(self.data.T,error)/self.m
            theta_i = theta_i - learning_rate*gradient
            cost = self.cost(theta_i,label_i_bin)
            cost_history.append(cost)
        return theta_i,cost_history

    def cost(self,theta_i,label_i_bin):
        hypothesis = sigmoid(np.dot(self.data,theta_i))
        cost = (-1/self.m)*(np.log(label_i_bin.dot(hypothesis)) + np.log((1-label_i_bin).dot(1-hypothesis)))
        return cost

    def classify(self):
        max_index = np.argmax(sigmoid(np.dot(self.data,self.theta)),axis = 1)
        return self.unique_labels[max_index].reshape(max_index.shape[0],1)


if __name__ == "__main__":
    # get iris_data
    iris = datasets.load_iris()
    data = iris.data
    features_names = iris.feature_names
    labels = iris.target
    # create logistic regression model
    model = LogisticRegression(data,labels)
    # train model
    cost_history = model.train()
    # plot cost_history
    for i in range(cost_history.shape[1]):
        plt.plot(range(1,1001),cost_history[:,i])
    plt.show()




