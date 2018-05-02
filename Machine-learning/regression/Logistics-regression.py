#逻辑回归

import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

def loadData(file,delimeter):
    data = np.loadtxt(file,delimiter=delimeter)
    # print('Dimensions:',data.shape)
    # print(data[1:6:])
    return data

def poltData(data,label_x,label_y,label_pos,label_neg,axes=None):
    neg = data[:,2] == 0
    pos = data[:,2] == 1
    # print(neg)

    if axes == None:
        axes = plt.gca()#使得axes成为当前子图
    axes.scatter(data[pos][:,0],data[pos][:,1],marker='+',c='k',s=60,linewidth=2,label=label_pos)
    axes.scatter(data[neg][:,0],data[neg][:,1],c='y',s=60,linewidth=2,label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True,fancybox=True)
    plt.show()

data = loadData('D:\data1.txt',',')

X = np.c_[np.ones((data.shape[0],1)),data[:,0:2]]#为每个数据前面加数字“1”，充当X0，
y = np.c_[data[:,2]]
# poltData(data,'Exam 1 score','Exam 2 score','Pass','Fail')

#定义sigmoid函数

def sigmoid(z):
    return (1.0 / (1 + np.exp(-z)))

def costFunction(theta,X,y):
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1 * (1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))#损失函数
    if np.isnan(J[0]):
        return (np.inf)
    return J[0]

#梯度下降
def gradient(theta,X,y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    grad = (1/m)*X.T.dot(h-y)
    return grad.flatten()


initial_theta = np.zeros(X.shape[1])
# cost = costFunction(initial_theta,X,y)
# grad = gradient(initial_theta,X,y)
# print('Cost:{}'.format(cost))
# print('Grad:{}'.format(grad))

#下面就是最小化损失函数,调用scipy里的最小化损失函数的minimize函数
res = minimize(costFunction,initial_theta,args=(X,y),method=None,jac=gradient,options={'maxiter':400})
# print(res.x)#表示最优的系数值

#下面是进行预测
def prediction(theta,X,threshold=0.5):
    P = sigmoid(X.dot(theta.T)) >= threshold
    return (P.astype('int'))

# print(sigmoid(np.array([1,45,85]).dot(res.x.T)))

# p = prediction(res.x,X)
# print('Train accuracy {}'.format(100*(sum(p == y.ravel())/p.size)))

#画决策边界
plt.scatter(45,85,s=60,c='r',marker='v',label='(45,85)')

x1_min,x1_max = X[:,1].min(),X[:,1].max()
x2_min,x2_max = X[:,2].min(),X[:,2].max()
xx1,xx2 = np.meshgrid(np.linspace(x1_min,x1_max),np.linspace(x2_min,x2_max))

h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)),xx1.ravel(),xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)
plt.contour(xx1,xx2,h,[0.5],linewidths=1,color='b')
poltData(data,'Exam 1 score','Exam 2 score','Pass','Fail')
plt.show()
