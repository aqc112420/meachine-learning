from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
import pandas as pd
from sklearn.datasets import make_classification
x,y = make_classification(n_samples=1000,n_features=2,n_redundant=0,n_informative=1,n_clusters_per_class=1)
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
x_train = x[:800]
x_test = x[800:]
y_train = y[:800]
y_test = y[800:]
# outputfile = "D:/1/1.xls"
# columns = ["x_train1","x_train2","x_test"]
# df = pd.DataFrame([x_train,y_train],index=range(len(x_train)),columns=columns)
print(x_train)

# positive_x1 = [x_test[i,0] for i in range(200) if y_test[i] == 1]
# positive_x2 = [x_test[i,1] for i in range(200) if y_test[i] == 1]
# negetive_x1 = [x_test[i,0] for i in range(200) if y_test[i] == 0]
# negetive_x2 = [x_test[i,1] for i in range(200) if y_test[i] == 0]
#
# from sklearn.linear_model import Perceptron
# clf = Perceptron(fit_intercept=False,n_iter=20,shuffle=False)
# clf.fit(x_train,y_train)
# # print(clf.coef_)
# # print(clf.intercept_)
#
# acc = clf.score(x_test,y_test)
# print(acc)
# plt.scatter(positive_x1,positive_x2,c='g',marker='o')
# plt.scatter(negetive_x1,negetive_x2,c='r',marker='x')
# line_x = np.arange(-4,4)
# line_y = line_x * (-clf.coef_[0][0] / clf.coef_[0][1]) - clf.intercept_
# plt.plot(line_x,line_y)
# plt.show()
#
