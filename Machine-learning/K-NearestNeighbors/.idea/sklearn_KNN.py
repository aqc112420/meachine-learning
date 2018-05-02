
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
n_neighbors =15
iris = load_iris()
X = iris.data[:,:2]#只取两个特征
y = iris.target
h = .02
#创建颜色地图
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])
for weights in ['uniform','distance']:
    clf = KNeighborsClassifier(n_neighbors,weights=weights)
    clf.fit(X,y)
    x_min,x_max = X[:,0].min() - 1,X[:,0].max() + 1
    y_min,y_max = X[:,1].min() - 1,X[:,1].max() + 1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
    plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.title('3-Class classification(k={},weights={}'.format(n_neighbors,weights))
plt.show()
