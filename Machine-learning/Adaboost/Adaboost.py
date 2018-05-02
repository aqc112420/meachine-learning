#bagging方法，是在原始数据集选择S次后得到S个新数据集的一种技术，新数据集和原数据集
#大小相等，每个数据集都是通过在原始的数据集中随机选择一个样本来进行替换而得到的
#替换意味着可以多次选择同一样本
#boosting也是一种分类器，下面是boosting的流行版本Adaboost（自适应boosting）


from numpy import *
#基于单层决策树（决策树桩）构建弱分类器


#建立简单的数据集
def loadSimpData():
    dataMat = matrix([[1,2.1],
                     [2,1.1],
                     [1.3,1],
                      [1,1],
                      [2,1],
                     ])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels


#单层决策树生成函数，dimen代表特征维数，threshVal代表特征的分类阈值，threshIneq代表分类不等号
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    #通过对阈值的比较来进行分类
    retArray = ones((shape(dataMatrix)[0],1))#将返回数组的元素全部置为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray#输出决策树桩标签

#遍历上一个桉树所有的可能的输入值，并找到数据集上
# 最佳的单层决策树（二叉树），最佳是根据数据权重向量D来定义的
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)#n代表数据有多少个特征
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps) + 1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals =\
                stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr#转换成单个数值
                # print("split:dim %d,thresh %.2f,thresh ineqal:\
                #  %s,the weighted error is %.3f"%(i,threshVal,\
                #                                  inequal,weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClassEst



#datMat,classLabels=loadSimpData()
# d=mat(ones((5,1))/5)
# bestStump,minError,bestClasEst=buildStump(datMat,classLabels,d)
# print (bestStump,minError,bestClasEst)

#基于单层决策树的Adaboost训练过程
#输入参数包括数据集，类别标签以及迭代次数
def adaboostTrainDS(dataArr,classLabels,numIt=40):#（DS表示单层决策树）
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)#开始权重被赋予了相同的值，D是一个概率分布向量，所有的元素之和为1
    aggClassEst = mat(zeros((m,1)))#记录每个数据点的类别估计累计值
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#首先是构建一个单层决策树，
        #bestStump包含了决策树的信息，error指的是分类的错误率，classEst指的是分类的预测结果
        print("D:",D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#alpha值代表了每个弱分类器所占的权重
        bestStump['alpha'] = alpha#将alpha的值添加到单层决策树中，该字典包括了
        #分类所需要的所有信息
        weakClassArr.append(bestStump)
        print("classEst:",classEst.T)
        #下面的三行是对权重D的调整公式
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst#不断累积预测的结果值
        print("aggClassEst:",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error:",errorRate,"\n")
        if errorRate == 0.0:
            break
    return weakClassArr


# classifierArray = adaboostTrainDS(datMat,classLabels,9)
# print(classifierArray)

#Adaboost分类函数
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)


datArr,labelArr = loadSimpData()
classifierArr = adaboostTrainDS(datArr,labelArr,30)
print(adaClassify([[0,0],[5,5]],classifierArr))
