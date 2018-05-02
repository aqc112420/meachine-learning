from numpy import *
import matplotlib.pyplot as plt
def loadDataSet(filename):#加载数据集，数据集的最后一列为label，前几列为数据，转化为float类型
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr)#将数据读入并保存为XMat矩阵
    yMat = mat(yArr).T#将label读入并保存为yMat矩阵
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)#根据公式W^ = (XTX)~ * XT*y
    return ws#返回最佳估计的线性回归系数


filename = 'D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch08/ex0.txt'
# xArr,yArr = loadDataSet(filename)
# ws = standRegres(xArr,yArr)
# xMat = mat(xArr)
# yMat = mat(yArr)
# yHat = xMat * ws
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
# xCopy = xMat.copy()
# xCopy.sort(0)
# yHat = xCopy * ws
# ax.plot(xCopy[:,1],yHat)
# plt.show()

#局部加权线性回归函数

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]#统计有多少个样本
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))#高斯核方法
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

#程序验证
# xArr,yArr = loadDataSet(filename)
# # print(yArr[0])
# # print(lwlr(xArr[0],xArr,yArr,1.0))
# yHat = lwlrTest(xArr,xArr,yArr,0.01)
# xMat = mat(xArr)
# srtInd = xMat[:,1].argsort(0)
# xSort = xMat[srtInd][:,0,:]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xSort[:,1],yHat[srtInd])
# ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='r')
# plt.show()

