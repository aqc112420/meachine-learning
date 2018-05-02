from Adaboost import *

def loadDataSet(fileName):#假定最后一列是类别标签
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


file ='D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch07/horseColicTraining2.txt'
datArr,labelArr = loadDataSet(file)

classifierArray = adaboostTrainDS(datArr,labelArr,10)
print(classifierArray)
# file1 ='D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch05/horseColicTest.txt'
#
# testArr,testLabelArr = loadDataSet(file1)
# prediction10 = adaClassify(testArr,classifierArray)
# errArr = mat(ones((67,1)))
# errArrSum = errArr[prediction10 != mat(testLabelArr).T].sum()
# print(errArrSum)

