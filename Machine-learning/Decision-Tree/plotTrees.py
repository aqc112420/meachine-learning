import matplotlib.pyplot as plt
decisionNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle='<-')#dict函数用来创造字典

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    #在图形中增加带箭头的注解。s表示要注解的字符串是什么，
    # xy对应箭头所在的位置，xytext对应文本所在位置，arrowprops定义显示字符串的属性
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',\
                            xytext=centerPt,textcoords='axes fraction',\
                            va='center',ha='center',bbox=nodeType,
                            arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()


#获取叶节点的数目和树的层数
def getNumLeafs(myTree):
    numLeafs = 0
    listKey = []
    for i in myTree.keys():
        listKey.append(i)
    firstStr = listKey[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    listKey = []
    for i in myTree.keys():
        listKey.append(i)
    firstStr = listKey[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


#构建plotTree函数

def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)


def plotTree(myTree,parentPt,nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    listKey = []
    for i in myTree.keys():
        listKey.append(i)
    firstStr = listKey[0]
    cntrPt = (plotTree.x0ff + (1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.y0ff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondeDict = myTree[firstStr]
    plotTree.y0ff = plotTree.y0ff - 1.0/plotTree.totalD
    for key in secondeDict.keys():
        if type(secondeDict[key]).__name__ == 'dict':
            plotTree(secondeDict[key],cntrPt,str(key))
        else:
            plotTree.x0ff = plotTree.x0ff + 1.0/plotTree.totalW
            plotNode(secondeDict[key],(plotTree.x0ff,plotTree.y0ff),cntrPt,leafNode)
            plotMidText((plotTree.x0ff,plotTree.y0ff),cntrPt,str(key))

    plotTree.y0ff = plotTree.y0ff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.x0ff = -0.5/plotTree.totalW;plotTree.y0ff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()


def retrieveTree(i):
    listOfTrees = [{'no surfacing':{0:'no',1:{'flippers':\
                                                  {0:'no',1:'yes'}}}},
                   {'no surfacing':{0:'no',1:{'flippers':\
                                                  {0:{'head':{0:'no',\
                                                              1:'yes'}},1:'no'}}}}
                   ]
    return listOfTrees[i]



myTree = retrieveTree(0)
createPlot(myTree)




