from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import lfw
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

a = 'D:\python_machine-learning\DataSets\lfw_home\lfw-funneled'
b = lfw.check_fetch_lfw(a)
print(b)

lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.4,funneled=False)
n_samples,h,w = lfw_people.images.shape
X = lfw_people.data
n_features = X.shape[1]

y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]#有多少个人进行人脸识别

print('n_samples:',n_samples)
print('n_features:',n_features)
print('n_classes:',n_classes)

#将数据划分为训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


#接下来进行降维的处理
n_components = 150
print('Extracting the top {} eigenfaces'\
      'from {} faces'.format(n_components,X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components,whiten=True ).fit(X_train)
print('done in {}'.format(time() - t0))
eigenfaces = pca.components_.reshape(n_components,h,w)
print('Projecting the input data on the eigenfaces'\
      'orthonormal basis')
t0 = time()
#进行降维工作
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print('done in {}'.format(time() - t0))

print('Fitting the classifier to the training set')
t0 = time()
param_grid = {'C':[1e3,5e3,1e4,5e4,1e5],
              'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1]}
clf = GridSearchCV(SVC(kernel='rbf',class_weight='auto'),param_grid)
clf = clf.fit(X_train_pca,y_trian)
print('done in {}'.format(time() - t0))
print('Best estimator found by grid search:')
print(clf.best_estimator_)


print("Predicting people's names on the test set ")
t0 = time()
y_pred = clf.predict(X_test_pca)
print('done in {}'.format(time() - t0))

print(classification_report(y_test,y_pred,target_names=target_names))
print(confusion_matrix(y_test,y_pred,labels=range(n_classes)))


def plot_gallery(images,titles,h,w,n_row=3,n_col=4):
    plt.figure(figsize=(1.8*n_col,2.4*n_row))
    plt.subplots_adjust(bottom=0,left=.01,right=0.99,top=0.9,hsapce=0.35)
    for i in range(n_row*n_col):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(images[i].reshpe((h,w)),cmap=plt.cm.gray)
        plt.title(titles[i],size=12)
        plt.xticks(())
        plt.yticks(())


def title(y_pred,y_test,target_names,i):
    pred_name = target_names[y_pred[i]].rsplit(' ',1)[-1]
    ture_name = target_names[y_test[i]],rsplit(' ',1)[-1]
    return 'predicted: {}\nture: {}'.format(pred_name,ture_name)

prediction_titles = [title(y_pred,y_test,target_names,i) for i in  range(y_pred.shape[0])]
plot_gallery(X_test,prediction_titles,h,w)

eigenface_titles = ["eigenface {}".format(i for i in range(eigenfaces.shape[0]))]
plot_gallery(eigenfaces,eigenface_titles,h,w)
plt.show()