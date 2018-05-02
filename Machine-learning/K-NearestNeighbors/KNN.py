import numpy as np
from numpy import *
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
import operator
def createDateSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels


def classify0(inX,dataSet,labels,k):
    dataSetSize =  dataSet.shape[0]#计算数据集的样本量
    diffMat = tile(inX,(dataSetSize,1)) - dataSet#用tile函数将inX重复dataSetSize行，再减去dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)#每行求和，相当于对每个样本差进行求和
    distances = sqDistances**0.5#开方

    sortedDistIndicies = distances.argsort()#按照升序将距离排列
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1#如果距离相同就加一，get(voteIlabel,0)将默认值改为0
    sortedClassCount = sorted(classCount.items(),key=\
operator.itemgetter(1),reverse=True)#operator.itemgetter(1)意思是使用value值的大小进行排序
    return sortedClassCount[0][0]

# group,labels = createDateSet()
# print(classify0([0,0],group,labels,3))


# def file2matrix(filename):#将数据集读进来，并进行转化
#     f = open(filename)
#     arrayOlines = f.readlines()
#     numberOfLines = len(arrayOlines)
#     returnMat = zeros((numberOfLines, 3))
#     classLabelVector = []
#     index = 0
#     for line in arrayOlines:
#         line = line.strip()
#         listFromLine = line.split('\t')
#         returnMat[index, :] = listFromLine[0:3]
#         if listFromLine[-1] == 'largeDoses':
#             classLabelVector.append((3))
#         elif listFromLine[-1] == 'smallDoses':
#             classLabelVector.append((2))
#         elif listFromLine[-1] == 'didntLike':
#             classLabelVector.append((1))
#         index += 1
#     return returnMat, classLabelVector

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


#数值归一化处理是因为三个特征权重相同，而飞机里程数值偏大，会对于其他两个特征造成影响，因此要进行归一化处理
#s数值归一化，按照公式newValues= （oldValues- min）/max -min
def autoNorm(dataSet):
    minVals = dataSet.min(0)#取出每一列当中的最小值放在变量minVals中
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet / tile(ranges,(m,1))
    return normDataSet,ranges,minVals
# #程序测试代码
# def datingClassTest():
#     hoRato = 0.10
#     datingDataMat,datingLabels = file2matrix('D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch02/datingTestSet.txt')
#     normMat,ranges,minVals = autoNorm(datingDataMat)
#     m = normMat.shape[0]
#     numTestVecs = int(m*hoRato)#选取多少个样本用于测试
#     errorCount = 0.0
#     for i in range(numTestVecs):
#         classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
#                                      datingLabels[numTestVecs:m],4)
#         print("The classifier came back with:{},the real answer is:{}\
#         ".format(classifierResult,datingLabels[i]))
#         if classifierResult != datingLabels[i]:
#             errorCount += 1.0
#     print("The total error rate is:{}".format(errorCount/float(numTestVecs)))
#

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing"
                              "video games?\t"))
    ffMiles = float(input("frequent flier miles earned per year?\t"))
    iceCream = float(input("liters of ice cream consumed per year?\t"))
    datingDataMat,datingLabels = file2matrix('D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch02/datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:",resultList[classifierResult - 1])


# classifyPerson()

def img2vector(filename):
    returnVect = zeros((1,1024))
    f = open(filename)
    for i in range(32):
        lineStr = f.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#testVector = img2vector("D:\python_deep learning\机器学习\机器学习实战源代码\machinelearninginaction\Ch02/trainingDigits/0_0.txt")

#
def handwritingClassTest():
    hwlabels = []
    trainingAdrss = 'D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch02/trainingDigits'
    trainingFlieList = listdir(trainingAdrss)#返回文件夹下的文件名称
    m = len(trainingFlieList)#返回一共有多少个文件
    #print(trainingFlieList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFlieList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split('_')[0])
        hwlabels.append(classNumStr)#获取数据的标签
        trainingMat[i,:] = img2vector(trainingAdrss+'/'+fileNameStr)
        testAdrss = 'D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch02/testDigits'
        testFileList = listdir(testAdrss)
    errorCount =0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])#获取真实的标签
        vectorUnderTest = img2vector(testAdrss+"/{}".format(fileNameStr))
        # print(vectorUnderTest.shape)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwlabels,3)
        print("The classifier came back with {},the real answer is:\
            {}".format(classifierResult,classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nThe total number of errors is: {}".format(errorCount))
    print("\nThe total error rate is: {}".format(errorCount/float(mTest)))


handwritingClassTest()