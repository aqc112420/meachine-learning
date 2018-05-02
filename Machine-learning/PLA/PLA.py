import matplotlib.pyplot as plt
from numpy import *
import operator
import time


def createTrainDataSet():  #  训练样本,第一个1为阈值对应的w，下同
    trainData = [[1,1,4],
                 [1,2,3],
                 [1,-2,3],
                 [1,-2,2],
                 [1,0,1],
                 [1,1,2]]
    label = [1,1,1,-1,-1,-1]
    return trainData,label


def createTestDataSet():
    testData = [[1,1,1],
                [1,2,0],
                [1,2,4],
                [1,1,3]]
    return testData


def sigmoid(X):
    X = float(X)
    if X>0:
        return 1
    elif X<0:
        return -1
    else:
        return 0


def PLA(trainDataIn,trainLabelIn):
    traindata = mat(trainDataIn)
    trainlabel = mat(trainLabelIn).transpose()
    m,n = shape(traindata)
    W = ones((n,1))
    while True:
        iscompleted = True
        for i in range(m):
            if sigmoid(dot(traindata[i],W)) == trainlabel[i]:#判断按照此直线划分的类型是否跟该点的实际类型相同
                continue
            else:
                iscompleted = False
                W += (trainlabel[i]*traindata[i]).transpose()#W的更新方法为W += y*X
        if iscompleted:
            break

    return W#返回更新后的W值


def classify(inX,W):#验证W的结果
    result = sigmoid(sum(W*inX))
    if result > 0:
        return 1
    else:
        return -1


def plotBestFit(W):
    traindata,label = createTrainDataSet()
    dataArr = array(traindata)
    n = shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(label[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)#以下三行代码是为了画出分割曲线
    y = (-W[0]-W[1]*x)/W[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()


def classifyall(datatest,W):
    predict = []
    for data in datatest:
        result = classify(data,W)
        predict.append(result)
    return predict

def main():
    trainData,label = createTrainDataSet()
    testdata = createTestDataSet()
    W = PLA(trainData,label)
    result = classifyall(testdata,W)
    plotBestFit(W)
    print(W)
    print(result)

if __name__  == "__main__":
    start = time.clock()
    main()
    end = time.clock()
    print((end - start))

