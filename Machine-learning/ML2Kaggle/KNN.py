#利用KNN对鸢尾花数据集进行分类

from sklearn.datasets import load_iris
iris = load_iris()

#将数据集分为75%训练集和25%的测试集
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=.25,random_state=33)

#使用KNN分类器对鸢尾花数据集进行类别预测
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_predict = knn.predict(X_test)

#对预测结果进行评价
print("The Accuracy of KNN Classifier is:",knn.score(X_test,y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=iris.target_names))



