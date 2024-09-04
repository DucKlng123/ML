import numpy as np
from pyexpat import features

from numpy import number

from utils.features import prepare_for_training

class LinearRegression:
    def __init__(self, data,labels,polynomial_degree = 0, sinusoid_degree = 0, normalize_data = True):
        """
        数据预处理
        得到特征个数并初始化参数矩阵
        """
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree = 0 , sinusoid_degree = 0, normalize_data= True)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features,1))

    def train(self,alpha,num_iterations = 500):
        """
        训练模型
        """
        cost_history = self.gradient_descent(alpha,num_iterations)
        return self.theta,cost_history

    def gradient_descent(self, alpha, num_iterations = 500):
        cost_history = []

        for i in range(num_iterations) :
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.theta))
        return cost_history
    def gradient_step(self, alpha):
        """
        梯度下降一步
        """
        num_examples = self.data.shape[0]
        prediction = self.hypothesis(self.data,self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - (alpha/num_examples) * np.dot(self.data.T,delta)
        self.theta = theta


    @staticmethod
    def hypothesis(data,theta):
        """
        预测函数
        """
        prediction = np.dot(data,theta)
        return prediction

    def cost_function(self,data,theta):
        """
        代价函数
        """
        prediction = self.hypothesis(data,theta)
        delta = prediction - self.labels
        cost = np.dot(delta.T,delta)/(2*data.shape[0])
        return cost

    def get_cost(self,data,labels):
        data_processed = prepare_for_training(data, polynomial_degree = self.polynomial_degree , sinusoid_degree = self.sinusoid_degree, normalize_data= self.normalize_data)[0]
        return self.cost_function(data_processed,labels)

    def predict(self,data):
        data_processed = prepare_for_training(data, polynomial_degree = self.polynomial_degree , sinusoid_degree = self.sinusoid_degree, normalize_data= self.normalize_data)[0]
        predictions = self.hypothesis(data_processed,self.theta)

