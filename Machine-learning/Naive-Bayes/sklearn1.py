from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data,iris.target).predict(iris.data)
print(iris.data.shape[0],(iris.target != y_pred).sum())