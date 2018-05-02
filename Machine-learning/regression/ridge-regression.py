#当特征来的比数据多的时候，说明输入数据矩阵不再是满秩，此时不能用上述的线性回归或者是lwlr，
# 因为上述两种方法都要运用到求逆
#为了解决这一问题，运用岭回归即可
#公式：w = (XTX + λI)的逆 * XT*y
#岭回归也运用在估计中加入偏差，从而得到更好的估计，通过引入λ来限制所有的w之
# 和，通过引入该惩罚项，能够减少不重要的参数
#与前几章里训练其他参数所用的方法类似，这里通过预测误差最小化得到1 数据获取之后,
#首先抽一部分数据用于测试，剩余的作为训练集用于训练参数^训练完毕后在测试集上测试预
#测性能。通过选取不同的λ来重复上述测试过程，最终得到一个使预测误差最小的λ。

import matplotlib.pyplot as plt
from regression import *
from numpy import *
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):#在一组λ上测试结果
    #以下是数据的标准化过程，各自特征减去自己的均值，然后除以方差
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)#每一列求均值
    yMat = yMat - yMean#减去均值
    xMeans = mean(xMat,0)#求取XMat中各个特征的均值
    xVar = var(xMat,0)#求取每个特征下的方差
    xMat = (xMat - xMeans)/xVar
    #接下来在30个不同的λ下调用ridgeRegres函数
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat


#测试；岭回归函数

fileName = 'D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch08/abalone.txt'
abX,abY = loadDataSet(fileName)
ridgeWeights = ridgeTest(abX,abY)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()
