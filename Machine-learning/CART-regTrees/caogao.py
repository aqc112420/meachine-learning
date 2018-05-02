from numpy import *
#from regTrees import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

file = 'D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch09/ex00.txt'
myDat = loadDataSet(file)
myDat = mat(myDat)
print(var(myDat[:,-1])
)