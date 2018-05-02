from sklearn import svm
import numpy as np
import pylab as pl
np.random.seed(0)#每次运行程序都产生相同的值
X = np.r_[np.random.randn(20,2) - [2,2],np.random.randn(20,2) + [2,2]]

Y = [0] *20 + [1] * 20
clf = svm.SVC(kernel='linear')
clf.fit(X,Y)
w = clf.coef_[0]#得到训练出来的w的值
a = -w[0] / w[1]
xx = np.linspace(-5,5)
yy = a * xx - (clf.intercept_[0] / w[1])

b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])
print('w:',w)
print('a:',a)
print('support_vector_',clf.support_vectors_)
print('clf.coef_:',clf.coef_)
pl.plot(xx,yy,'k-')
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'k--')
pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80,facecolor='None')
pl.scatter(X[:,0],X[:,1],c='yellow',cmap=pl.cm.Paired)
pl.axis('tight')
pl.show()