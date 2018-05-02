from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.datasets import load_iris
from numpy import *
iris = load_iris()
data = iris.data
label = iris.target

error = 0.0
irisIndex = [1,3,7,10,45,76,87,1,120,147]
clf = RFC(n_estimators=10)
clf.fit(data,label)
for index in irisIndex:
    if label[index] != clf.predict(data[index].reshape(1,-1)):
        error += 1

irisLen = len(irisIndex)
print("The total accrual is:{}".format((irisLen-error) / irisLen))

# print(clf.predict(data[0].reshape(1,-1)))