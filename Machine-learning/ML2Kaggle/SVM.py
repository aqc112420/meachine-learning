#利用SVM来对手写字进行识别

#导入数据集
from sklearn.datasets import load_digits
digits = load_digits()
# print(digits.data.shape)
#将手写体数据进行分割
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=.25,random_state=33)

#SVM进行手写字识别
from sklearn.preprocessing import StandardScaler
#从sklearn.svm里导入基于线性假设的SVM分类器LinearSVC
from sklearn.svm import LinearSVC

#仍然需要对训练和测试的特征数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lsvc = LinearSVC()
lsvc.fit(X_train,y_train)
y_predict = lsvc.predict(X_test)

#使用模型自带的评估函数进行准确率的预测
print("The Accuracy of Linear SVC is:",lsvc.score(X_test,y_test))

#利用classification_report模块来获得SGDClassifier其他三个指标的结果
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=digits.target_names.astype(str)))
