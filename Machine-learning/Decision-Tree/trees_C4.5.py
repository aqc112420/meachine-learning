from math import log2
import operator

#运用C4.5算法,C4.5算法采用的是增益率来选取最终的特征。


#计算信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log2(prob)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#按照给定的特征划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# myDat,labesl = createDataSet()
# a = splitDataSet(myDat,0,1)
# b = splitDataSet(myDat,0,0)
# print(a)

#循环遍历整个数据集，循环计算信息熵和splitDataSet（）函数
#找到最好的特征划分方式

def chooseBestFeatureToSplit(dataSet):#利用信息相对增益的大小来选取特征
    numFeatures = len(dataSet[0]) - 1#每一列代表一个特征，减去最后一个结果列
    baseEntropy = calcShannonEnt(dataSet)#计算原始信息熵
    bestGainRate = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]#取第i个属性的值
        uniqueVals = set(featList)#得到该属性下不重复的取值
        newEntropy = 0.0
        entroryRate = 0.0
        splitE = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))#计算value下的概率
            newEntropy += prob * calcShannonEnt(subDataSet)
            splitE -= prob * log2(prob)#计算prob*log2（prob）
        infoGain = baseEntropy - newEntropy
        entroryRate = infoGain/splitE

        if (entroryRate > bestGainRate):
            bestGainRate = entroryRate
            bestFeature = i
    return bestFeature

#多数表决的方法来决定叶子节点的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] +=1
    sortedClassCount = sorted(classCount.items(),\
    key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#创建决策树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]#将标签的种类都放到classList里
    if classList.count(classList[0]) == len(classList):#第一个停止条件：如果classList中所有的标签都跟classList相同，直接返回classList
        return classList[0]
    if len(dataSet[0]) == 1:#第二个停止条件：如果使用完所有的特征，仍不能将数据集划分成仅包含唯一类别的分组\
        return majorityCnt(classList)#那么就用上面的多数表决函数来选择出现次数最多的类别来作为返回值
    bestFeat = chooseBestFeatureToSplit(dataSet)#最好的特征放在bestFeat里
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]#为了不改变原标签labels里面的内容，使用subLabels代替
        myTree[bestFeatLabel][value] = createTree(splitDataSet\
                                                      (dataSet,\
                                                       bestFeat,\
                                                       value),subLabels)
    return myTree

#测试决策树
# myDat,labels = createDataSet()
myDat = [['青绿','蜷缩','浊响','清晰','凹陷','硬滑','是'],
         ['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑','是'],
         ['乌黑','蜷缩','浊响','清晰','凹陷','硬滑','是'],
         ['青绿','蜷缩','沉闷','清晰','凹陷','硬滑','是'],
         ['浅白','蜷缩','浊响','清晰','凹陷','硬滑','是'],
         ['青绿','稍蜷','浊响','清晰','凹陷','软粘','是'],
         ['乌黑','稍蜷','浊响','稍糊','稍凹','软粘','是'],
         ['乌黑','稍蜷','浊响','清晰','稍凹','硬滑','是'],
         ['乌黑','稍蜷','沉闷','稍糊','稍凹','硬滑','否'],
         ['青绿','硬挺','清脆','清晰','平坦','软粘','否'],
         ['浅白','硬挺','清脆','模糊','平坦','硬滑','否'],
         ['浅白','蜷缩','浊响','模糊','平坦','软粘','否'],
         ['青绿','稍蜷','浊响','稍糊','凹陷','硬滑','否'],
         ['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑','否'],
         ['乌黑','稍蜷','浊响','清晰','稍凹','软粘','否'],
         ['浅白','蜷缩','浊响','模糊','平坦','硬滑','否'],
         ['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑','否']
         ]
labels = ['色泽','根蒂','敲声','纹理','脐部','触感']
myTree = createTree(myDat,labels)
print(myTree)
