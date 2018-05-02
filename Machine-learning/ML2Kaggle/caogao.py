
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split#用于分割数据集

column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size',
                'Unifromity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size',
                'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class'
                                                               ]

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                   names=column_names)

data = data.replace(to_replace='?',value=np.nan)

data = data.dropna(how='any')

X_train,X_test,y_train,y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],
                                                 test_size=0.25,random_state=33)

# print(y_train.value_counts())
# print(y_test.value_counts())

from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

#标准化数据，保证每个维度的特征数据方差为1，均值为0，
# 使得预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#初始化LogisticRegression与SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier()
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)

sgdc.fit(X_train,y_train)
sgdc_y_predict = sgdc.predict(X_test)

#使用下列代码来进行预测任务的性能分析

from sklearn.metrics import classification_report
print("Accuracy of LR Classifier:",lr.score(X_test,y_test))
#利用classification_report模块来获得LogisticRegression其他三个指标的结果
print(classification_report(y_test,lr_y_predict,target_names=['Bengin','Malignant']))


print("Accuracy of SGD Classifier:",sgdc.score(X_test,y_test))
#利用classification_report模块来获得SGDClassifier其他三个指标的结果
print(classification_report(y_test,sgdc_y_predict,target_names=['Benign','Malignant']))

