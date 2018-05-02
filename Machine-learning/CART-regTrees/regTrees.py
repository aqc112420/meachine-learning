from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]#这里的nonzero[0]返回的是非零元素所在的行列表
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1


#树构建函数

# testMat = mat(eye(4))
# mat0,mat1 = binSplitDataSet(testMat,1,0.5)
# print(mat0)
# print(mat1)


#构建chooseBestSplit函数，来完成用最佳方式切分数据集并生成相应的叶节点
#其中leafType是对创建叶节点的函数的引用，errType是对总方差计算函数的引用，ops是一个用户定义的
#参数有成的元组，用以完成树的构建


#在回归树中，叶节点的模型就是目标变量的均值
def regLeaf(dataSet):
    return mean(dataSet[:,-1])


def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]


def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0]#容许的误差下降值
    tolN = ops[1]#切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:#倒数最后一列转化为set，
        # 统计不同剩余特征值的数目
        #如果为1，就不需要切分而直接返回
        return None,leafType(dataSet)
    m,n = shape(dataSet)#数据集的大小，n表示有多少列
    S = errType(dataSet)#最好的特征通过计算平方总误差
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):#表示有多少个特征
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):#遍历每一个特征下的不同取值
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)#将数据分为两份
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue#如果分完后数据子集的大小小于tolN，那么就不应该分
            newS = errType(mat0) + errType(mat1)#计算分完之后的总方差
            if newS <bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:#如果切分数据集之后效果提升不够大，那么就不应该进行切分操作而是直接
        #创建叶节点
        return None,leafType(dataSet)

    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)#在最好的切分特征和特征值下将数据切分
    if (shape(mat0)[0] <tolN) or (shape(mat1)[0] < tolN):
        return None,leafType(dataSet)#如果分完后数据子集的大小小于tolN，那么就不应该分，而是直接创建叶节点
    return bestIndex,bestValue


def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(10000,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree


# file = 'D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch09/ex2.txt'
# myDat = loadDataSet(file)
# myDat = mat(myDat)
# print(createTree(myDat))

# import matplotlib.pyplot as plt
# plt.plot(myDat[:,0],myDat[:,1],'ro')
# plt.plot(0.488130,-0.097791,'b^')
# plt.show()


#后剪枝
def isTree(obj):#用于测试输入变量是否是一棵树
    return (type(obj).__name__=='dict')


def getMean(tree):#从上往下遍历直到叶节点为止，如果
    # 找到两个叶节点就计算他们的平均值
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right']) / 2.0


def prune(tree,testData):
    if shape(testData)[0] ==0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge =sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

#
#
file2 = 'D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch09/ex2.txt'
myDat2 = loadDataSet(file2)
myMat2 = mat(myDat2)
myTree = createTree(myMat2,ops=(0,1))
print(myTree)
testFile =  'D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch09/ex2test.txt'
myDat2Test = loadDataSet(testFile)
myMat2Test = mat(myDat2Test)
print('\n\n')
print(prune(myTree,myMat2Test))



