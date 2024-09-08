import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.distutils.misc_util import yellow_text
import plotly
import plotly.graph_objs as go

from linearRegression import LinearRegression


#plot
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR

data = pd.read_csv('../data/world-happiness-report-2017.csv')
print (data.head())
#得到训练和测试数据
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name_1 = 'Economy..GDP.per.Capita.'
input_param_name_2 = 'Freedom'
output_param_name = 'Happiness.Score'

x_train =  train_data[[input_param_name_1,input_param_name_2]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name_1,input_param_name_2]].values
y_test = test_data[[output_param_name]].values





#训练
num_iterations = 500
learning_rate = 0.01
polynomial_degeree = 15
sinosoid_degree = 0
normalize_data = True
LR = LinearRegression(x_train,y_train,polynomial_degeree,sinosoid_degree,normalize_data)
theta, cost_history = LR.train(learning_rate,num_iterations)
print ('开始的损失',cost_history[0])
print ('最后的损失',cost_history[-1])

print('Theta: ', theta)
print('num_features', LR.data.shape[1])

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost history')
plt.show()

# #draw 3D plot
# mesh_size = .02
# margin = 0
#
# train_data = train_data.rename(columns={'Economy..GDP.per.Capita.': 'Economy'})
# train_data = train_data.rename(columns={'Happiness.Score': 'Happiness'})
# X = train_data[['Economy', 'Freedom']]
# y = train_data['Happiness']
#
# # Condition the model on sepal width and length, predict the petal width
# model = SVR(C=1.)
# model.fit(X, y)
#
# # Create a mesh grid on which we will run our model
# x_min, x_max = X.Economy.min() - margin, X.Economy.max() + margin
# y_min, y_max = X.Freedom.min() - margin, X.Freedom.max() + margin
# xrange = np.arange(x_min, x_max, mesh_size)
# yrange = np.arange(y_min, y_max, mesh_size)
# xx, yy = np.meshgrid(xrange, yrange)
#
# # Run model
# pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
# pred = pred.reshape(xx.shape)
#
# # Generate the plot
# fig = px.scatter_3d(train_data, x='Economy', y='Freedom', z='Happiness')
# fig.update_traces(marker=dict(size=5))
# fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
# fig.show()

