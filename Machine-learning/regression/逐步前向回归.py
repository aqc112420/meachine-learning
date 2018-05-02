from numpy import *
from regression import *

def rssError(yArr,yHatArr):
    return ((yArr - yHatArr)**2).sum()


def regularize(xMat):#实现数据的标准化，满足每一列（即每个特征）均值为零，单位方差
    inMat = xMat.copy()
    inMeans = mean(inMat,0)
    inVar = var(inMat,0)
    inMat = (inMat - inMeans)/inVar
    return inMat


def stageWise(xArr,yArr,eps=0.01,numIt=100):#eps是每次迭代的步长
    #numIt表示迭代次数
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt,n))#每一行代表迭代一次所得到的权重ws
    ws = zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):#计算每一个权重的改变
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)#计算损失
                if rssE < lowestError:#如果损失减小
                    lowestError = rssE#最小损失就是rssE
                    wsMax = wsTest#最佳的权重就是wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

fileName = 'D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch08/abalone.txt'

xArr,yArr = loadDataSet(fileName)
print(stageWise(xArr,yArr,0.001,5000))

xMat = mat(xArr)
yMat = mat(yArr)
xMat = regularize(xMat)
yM = mean(yMat,0)
yMat = yMat - yM
weights = standRegres(xMat,yMat.T)
print(weights.T)

