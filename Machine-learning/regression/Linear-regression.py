#手写线性回归

# import numpy as np
# import pylab
#
#
# def Linear_regression(file):
#     data = np.loadtxt(file,delimiter=',')
#     learning_rate = 0.001
#     initial_b = 1.0
#     initial_m = 0.0
#     num_iter = 1000
#
#     print('initial variables:\n initial_b={}\n initial_m={}\n error of begin={}\n'\
#           .format(initial_b,initial_m,compute_error(initial_b,initial_m,data)))
#     [b,m] = optimizer(data,initial_b,initial_m,learning_rate,num_iter)
#     print('final formula parameters:\n b = {}\n m={}\n error of end = {}\n'\
#           .format(num_iter,b,m,compute_error(b,m,data)))
#     plot_data(data,b,m)
#
#
# def compute_gradient(b_current,m_current,data,learning_rate):
#     b_gradient = 0
#     m_gradient = 0
#     N = float(len(data))
#
#     for i in range(len(data)):
#         x = data[i,0]
#         y = data[i,1]
#
#         b_gradient += -(2/N)*(y-((m_current*x)+b_current))
#         m_gradient += -(2/N)*x*(y-((m_current*x)+b_current))
#
#     new_b = b_current - (learning_rate * b_gradient)
#     new_m = m_current - (learning_rate * m_gradient)
#     return [new_b,new_m]
#
#
# def compute_error(b,m,data):
#     totalError = 0.0
#     x = data[:,0]
#     y = data[:,1]
#     totalError = (y - (x*m+b))**2
#     totalError = np.sum(totalError,axis=0)
#     return totalError/float(len(data))
#
#
# def optimizer(data,starting_b,starting_m,learning_rate,num_iter):
#     b = starting_b
#     m = starting_m
#     for i in range(num_iter):
#         b,m = compute_gradient(b,m,data,learning_rate)
#         if i % 100 == 0:
#             print('iter{}:error={}'.format(i,compute_error(b,m,data)))
#     return [b,m]
#
#
# def plot_data(data,b,m):
#     x = data[:,0]
#     y = data[:,1]
#     y_predict = m*x+b
#     pylab.plot(x,y,'o')
#     pylab.plot(x,y_predict,'k-')
#     pylab.show()
#
#
# Linear_regression('D:\python_machine-learning\DataSets\ML/data.csv')

#基于tensorflow实现线性回归

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = 200
x = np.linspace(-1,1,N)
y = 2.0*x + np.random.standard_normal(x.shape)*0.3 +0.5
x = x.reshape([N,1])
y = y.reshape([N,1])

# plt.scatter(x,y)
# plt.plot(x,x*2+0.5)
# plt.show()

#建图
inputx = tf.placeholder(dtype=tf.float32,shape=[None,1])
groundY = tf.placeholder(dtype=tf.float32,shape=[None,1])
W = tf.Variable(tf.random_normal([1,1],stddev=0.01))
b = tf.Variable(tf.random_normal([1],stddev=0.01))
pred = tf.matmul(inputx,W) + b
loss = tf.reduce_sum(tf.pow(pred-groundY,2))

#优化目标函数
train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#加入监控点
tf.summary.scalar('loss',loss)
merged = tf.summary.merge_all()

#初始化所有变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    writer = tf.summary.FileWriter("D:/log/",sess.graph)

    sess.run(init)
    for i in range(20):
        sess.run(train,feed_dict={inputx:x,groundY:y})
        predArr,lossArr = sess.run([pred,loss],feed_dict={inputx:x,groundY:y})
        print(lossArr)

        summary_str = sess.run(merged,feed_dict={inputx:x,groundY:y})
        writer.add_summary(summary_str,i)
        WArr,bArr = sess.run([W,b])
        print(WArr,bArr)
        plt.scatter(x,y)
        plt.plot(x,WArr*x +bArr)
        plt.show()