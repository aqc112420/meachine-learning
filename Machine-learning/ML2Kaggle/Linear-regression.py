from sklearn.datasets import load_boston
boston = load_boston()

from sklearn.cross_validation import train_test_split
import numpy as np
X = boston.data
y = boston.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=33)

#由于数据目标房价之间差异较大，因此需要标准化处理
# print(np.max(boston.target))
# print(np.min(boston.target))
# print(np.mean(boston.target))

from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

#使用线性回归模型LinearRegression和SGDRegressor分别对波士顿房价进行预测
from sklearn.linear_model import LinearRegression,SGDRegressor
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_y_pred = lr.predict(X_test)

sgdr = SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr_y_pred = sgdr.predict(X_test)

#使用三种回归评价机制以及两种调用R-squared评价模块的方法，对本节模型的回归性能作出评价

print("The value of default measurement of LinearRegression is: ",lr.score(X_test,y_test))

from sklearn.metrics import r2_score,mean_squared_error,median_absolute_error

#使用r2_score模块输出评价结果
print("The value of R-squared of LinearRegression is: ",r2_score(y_test,lr_y_pred))

#使用mean_squared_error模块输出评价结果

print("The mean aquared error of LinearRegression is:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_pred)))

#使用median_absolute_error模块输出评价结果
print("The mean absolute error of LinearRegression is:",median_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_pred)))



print("The value of default measurement of SGDRegressor is: ",sgdr.score(X_test,y_test))


#使用r2_score模块输出评价结果
print("The value of R-squared of SGDRegressor is: ",r2_score(y_test,sgdr_y_pred))

#使用mean_squared_error模块输出评价结果

print("The mean aquared error of SGDRegressor is:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_pred)))

#使用median_absolute_error模块输出评价结果
print("The mean absolute error of SGDRegressor is:",median_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_pred)))


