import numpy as np
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
def createDateSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels


def classify0(inX,dataSet,labels,k):
    dataSetSize =  dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=\
operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    f = open(filename)
    arrayOlines = f.readlines()
    numberOfLines = len(arrayOlines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if listFromLine[-1] =='largeDoses':
            classLabelVector.append((3))
        elif listFromLine[-1] =='smallDoses':
            classLabelVector.append((2))
        elif listFromLine[-1] =='didntLike':
            classLabelVector.append((1))
        index += 1
    return returnMat,classLabelVector


#s数值归一化，按照公式newValues= （oldValues- min）/max -min
def autoNorm(dataSet):
    minVals = dataSet.min(0)#取出每一列当中的最小值放在变量minVals中
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet -np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals
#程序测试代码
def datingClassTest():
    hoRato = 0.10
    datingDataMat,datingLabels = file2matrix('D:\python_deep learning\机器学习\机器学习实战源代码\machinelearninginaction\Ch02/datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRato)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                     datingLabels[numTestVecs:m],4)
        print("The classifier came back with:{},the real answer is:{}\
        ".format(classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("The total error rate is:{}".format(errorCount/float(numTestVecs)))
# fig = plt.figure()
# ax =fig.add_subplot(111)
# ax.scatter(datingDataMat[:,0],datingDataMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
# plt.show()
def classifyPerson():
    resultList = ['not at all','in small doses','in large doese']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('D:\python_deep learning\机器学习\机器学习实战源代码\machinelearninginaction\Ch02/datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,\
                                 normMat,datingLabels,3)
    print("You will probably like this person:",\
          resultList[classifierResult -1])


classifyPerson()