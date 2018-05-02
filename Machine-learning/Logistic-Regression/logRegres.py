import math
from numpy import *
import random
import matplotlib.pyplot as plt
#Logistic回归梯度上升优化算法
def loadDataSet():
    dataMet = []
    labelMat = list()
    file = 'D:\python_deep learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch05'
    fr = open(file+'/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMet.append([1.0,float(lineArr[0]),float(lineArr[1])])#将X0的值设置为1.0
        labelMat.append(int(lineArr[2]))
    return dataMet,labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# dataArr,labelMat = loadDataSet()
# a = gradAscent(dataArr,labelMat)
# # print(a)

#画出决策边界


def plotBestFit(wei):
    import matplotlib.pyplot as plt
    #weights = wei.getA()#将矩阵转化为array，才能进行后续的操作
    weights =wei
    dataMat,labelMat = loadDataSet()#加载数据和标签
    dataArr = array(dataMat)#将数据矩阵转化为array
    n = shape(dataArr)[0]#求出dataArr有多少行
    xcord1 = []
    ycord1 =[]
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:#标签是1的数据放到xcord1，ycord1里面去
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])

        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='r',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='g')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# plotBestFit(a)

#随即梯度上升


def stocGradAscent0(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)#产生1行n列的矩阵
    print(type(weights))
    print(type(dataMatrix))
    dataArr = array(dataMatrix)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]

    return weights

#
# dataArr,labelMat = loadDataSet()
# weights = stocGradAscent0(array(dataArr),labelMat)
# plotBestFit(weights)

#改进随机梯度上升算法

def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)

    for j in range(numIter):
        dataIndex =list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01#在每次迭代的过程中调整alpha的值
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

# dataArr,labelMat = loadDataSet()
# weights = stocGradAscent1(array(dataArr),labelMat,numIter=20)
# plotBestFit(weights)
#预测马的生死


def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    trainFile = 'D:\python_deep learning\Machine-Learning/Code-ML-ShiZhan\machinelearninginaction\Ch05/horseColicTraining.txt'
    testFile = 'D:\python_deep learning\Machine-Learning/Code-ML-ShiZhan\machinelearninginaction\Ch05/horseColicTest.txt'
    frTrain = open(trainFile)
    frTest = open(testFile)
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print('The error rate of this test is :{}'.format(errorRate))
    return errorRate

def multiTest():
    numTests =10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('After {} iterations the average error rate is:{}'.format(numTests,errorSum/float(numTests)))


multiTest()