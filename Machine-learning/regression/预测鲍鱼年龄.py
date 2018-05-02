from numpy import *
from regression import *


def rssError(yArr,yHatArr):
    return ((yArr - yHatArr)**2).sum()


fileName = 'D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch08/abalone.txt'
abX,abY = loadDataSet(fileName)
yHat01 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
yHat1 = lwlrTest(abX[100:199],abX[:99],abY[:99],1)
yHat10 = lwlrTest(abX[100:199],abX[:99],abY[:99],10)
print(rssError(abY[100:199],yHat01.T))
print(rssError(abY[100:199],yHat1.T))
print(rssError(abY[100:199],yHat10.T))
