#利用决策树来分析泰坦尼克上的人是否生还

import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# print(titanic.info())

#由于数据集存在数值缺失的情况，所以进行以下处理

#首先我们选择pclass，age，sex这三个特征
X = titanic[['pclass','age','sex']]
y = titanic['survived']
# print(X.info())
#考虑到年龄的缺失，类别型数据转化为数值特征
X['age'].fillna(X['age'].mean(),inplace=True)
# print(X.info())

#接下来进行数据集的分割
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=33)

#s使用DictVectorizer（特征转换器），将类别型的特征都单独剥离出来，
# 独成一列特征，数值型的则保持不变。
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
# print(vec.feature_names_)
#同样要对测试数据集的特征进行转换
X_test = vec.transform(X_test.to_dict(orient='record'))

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
y_predict = tree.predict(X_test)
print(tree.score(X_test,y_test))

from sklearn.metrics import classification_report
print(classification_report(y_predict,y_test,target_names=['died','survived']))
