
#import numpy as np
from numpy import *
import random
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']
                   ]
    classVec = [0,1,0,1,0,1]#1代表侮辱性文字，0代表正常言论
    return postingList,classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)

    return list(vocabSet)


def setOfWords2Vec(vocabList,inputSet):#将单词转化为0（没出现）1（出现）。并返回returnVec大小的列表
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word:{} is not in my Vocabulary!".format(word))
    return returnVec


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)#总的文档数目
    numWords = len(trainMatrix[0])#计算每篇文档中有多少个单词
    pAbusive = sum(trainCategory)/float(numTrainDocs)#求出所有侮辱性文章所占的比例（P(C1)）
    p0Num = ones(numWords)#创建一个一行32列的全0矩阵,
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:#在侮辱性文档的条件下
            p1Num += trainMatrix[i]#将侮辱性文章里面的所有关键词加到p1Num里
            p1Denom += sum(trainMatrix[i])#关键词的个数求和
        else:#在非侮辱性文档的条件下：
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = log(p1Num/p1Denom)#求出每个词汇在（侮辱文档条件下）所占的比例,转换为log显示,以单个的单词数除以总的单词数目就求得（P（Wi|C1））
    p0Vect = log(p0Num/p0Denom)#求出每个词汇在（非侮辱文档条件下）所占的比例,转换为log显示
    return p0Vect,p1Vect,pAbusive

# listOposts,listClasses = loadDataSet()
# myVocabList = createVocabList(listOposts)
# print(listOposts)
# print(myVocabList)
# trainMat = []
# for postinDoc in listOposts:
#     trainMat.append(setOfWords2Vec(myVocabList,postinDoc))#将trainMat矩阵中每一行转化为
#     #与setOfWords2Vec返回列表的同等大小
# p0V, p1V, pAb = trainNB0(trainMat,listClasses)
#
# print(p1V)

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):#化为log后，减去的分母相同，可以忽略
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'clssified as :',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as ',classifyNB(thisDoc,p0V,p1V,pAb))


def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index[word]] += 1

    return returnVec


#文件解析及完整的垃圾邮件测试函数
def textParse(bigString):
    import re
    ma = re.compile(r'\w+')
    listOfTokens = ma.findall(bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]


def spamTest():
    file1 = 'D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch04\email\spam'
    file2 = 'D:\python_machine-learning\Machine-Learning\Code-ML-ShiZhan\machinelearninginaction\Ch04\email\ham'
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open(file1+'/{}.txt'.format(i)).read())#将返回的词列表给予wordList
        docList.append(wordList)
        fullText.extend(wordList)#存储所有的词
        classList.append(1)#代表是垃圾邮件
        wordList = textParse(open(file2+'/{}.txt'.format(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)#代表不是垃圾邮件
    vocabList = createVocabList(docList)#创建词列表
    trainingSet = list(range(50))#训练样本总数
    testSet = []#将测试集样本的索引放入testSet
    for i in range(10):#从总的50个训练样本中选取10个作为测试集
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])#将测试集索引从总样本中剔除
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))#查看上边的setOfWords2Vec函数理解
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:#检验朴素贝叶斯分类器的效果
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) !=\
            classList[docIndex]:
            errorCount +=1
            print("classification error", docList[docIndex])
    print('The error rate is:',float(errorCount)/len(testSet))
    return fullText,vocabList


spamTest()