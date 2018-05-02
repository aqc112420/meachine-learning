import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

#人工选取pclass，age和sex作为判别乘客是否能够生还的特征
X = titanic[['pclass','age','sex']]
y = titanic['survived']


#对于缺失的年龄信息，我们使用全体乘客的平均年龄代替，这样可以保证顺利训练模型的同时，尽可能的不影响预测任务
X['age'].fillna(X['age'].mean(),inplace=True)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=33)

#对类别型特征进行转化，成为特征向量
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

#使用单一决策树进行模型训练及预测分析
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
tree_y_pred = tree.predict(X_test)

#使用随机森林分类器进行集成模型的训练及预测分析
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred = rfc.predict(X_test)

#使用梯度提升决策树进行集成模型的训练及预测分析
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred = gbc.predict(X_test)

from sklearn.metrics import classification_report

#输出单一决策树的评价结果
print("The Accuracy of decision tree is:",tree.score(X_test,y_test))
print(classification_report(tree_y_pred,y_test))

#输出随机森林的评价结果
print("The Accuracy of RF is:",rfc.score(X_test,y_test))
print(classification_report(rfc_y_pred,y_test))

#输出梯度提升决策树的评价结果
print("The Accuracy of GBC is:",gbc.score(X_test,y_test))
print(classification_report(gbc_y_pred,y_test))

