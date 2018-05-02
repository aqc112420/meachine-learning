# from Adaboost import *
from numpy import *
from sklearn.ensemble import AdaBoostClassifier
def loadSimpData():
    dataMat = matrix([[1,2.1],
                     [2,1.1],
                     [1.3,1],
                      [1,1],
                      [2,1],
                     ])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

datMat,labelMat = loadSimpData()

clf = AdaBoostClassifier(n_estimators=10)
clf.fit(datMat,labelMat)
print(clf.predict([[0,0],[5,5]]))
