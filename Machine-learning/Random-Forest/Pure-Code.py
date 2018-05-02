#参考文章  http://blog.csdn.net/flying_sfeng/article/details/64133822

#集成学习力个体学习器应该“好而不同”，即个体学习器要有一定的“准确性”，而且要有“多样性”，
#即学习器不能太差，而且彼此之间要有差异
#Boosting 个体学习器之间存在强依赖关系，必须串行生成序列化方法
#bagging和随机森林  个体学习器之间不存在强依赖关系，可同时生成并行化方法

'''
在我的实验中，使用“自助采样法”：给定包含m个样本的数据集，我们先随机取出一个样本放入
采样集中，再把该样本放回初始数据集，使得下次采样时该样本仍有可能被选中，这样，经过m次
随机操作，我们得到含m个样本的采样集，初始训练集中有的样本在采样集里多次出现，
有的则从未出现。按照这种方法，我们可以采样出T个含m个训练样本的采样集，
然后基于每个采样集训练处一个基学习器，再将这些基学习器进行结合，这就是Bagging的基本流程。
在对预测输出进行结合时，Bagging通常对分类任务使用简单投票法，对回归任务使用简单平均法。

'''
import csv
from random import randrange
import random
#1 导入文件并将所有特征转换为float形式
#加载数据
def loadCSV(filename):
    dataSet = []
    with open(filename,'r') as file:
        csvReader = csv.reader(file)
        for line in csvReader:
            # print(type(line))
            dataSet.append(line)
    return dataSet

#除了判别列，其他列均转换为float类型
def column_to_float(dataSet):
    featLen = len(dataSet[0])-1#判断需要有多少列需要转化
    for data in dataSet:
        for column in range(featLen):
            data[column] = float(data[column].strip())#strip()返回移除字符串头尾指定的字符生成的新字符串。


#2.将数据集分为n份，方便交叉验证

#将数据集分成N块，方便交叉验证
#将数据集dataset分成n_flods份，每份包含len(dataset) / n_folds个值，
# 每个值由dataset数据集的内容随机产生，每个值被使用一次
def splitDatset(dataSet,n_folds):
    fold_size = int(len(dataSet)/n_folds)#每一份所含有的数据的量
    dataSet_copy = list(dataSet)#复制一份dataSet，防止数据被串改
    dataSet_split = []
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(dataSet_copy))
            fold.append(dataSet_copy.pop(index))#将选中的数据放到fold里面，然后将这个数据从dataSet_copy中去除，保证每个样本只使用一次
        dataSet_split.append(fold)
    return dataSet_split#返回的数据是包含了n_folds个列表的大列表

#分割数据集
def data_split(dataSet,index,value):#这种分类的方式适合数值型二叉树，具体的问题应该改变分类方式
    left=[]
    right=[]
    for row in dataSet:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left,right

#计算分割代价，利用的是基尼指数 基尼指数（基尼不纯度）= 样本被选中的概率 * 样本被分错的概率
def split_loss(left,right,class_values):
    loss=0.0
    for class_value in class_values:
        left_size=len(left)
        if left_size!=0:  #防止除数为零
            prop=[row[-1] for row in left].count(class_value)/float(left_size)
            loss += (prop*(1.0-prop))
        right_size=len(right)
        if right_size!=0:
            prop=[row[-1] for row in right].count(class_value)/float(right_size)
            loss += (prop*(1.0-prop))
    return loss

#3 构造数据子集（随机采样），并在指定特征个数（假设m个，手动调参）
#下选取最优特征
#构造数据子集
def get_subsample(dataSet,ratio):
    subdataSet = []
    lenSubdata = round(len(dataSet)*ratio)
    while len(subdataSet) < lenSubdata:
        index = randrange(len(dataSet) - 1)
        subdataSet.append(dataSet[index])
    return subdataSet

#选取任意的n个特征，在这n个特征中，选取分割时的最优特征
def get_best_split(dataSet,n_features):
    features = []
    class_values = list(set(row[-1] for row in dataSet))#计算到底有多少类
    b_index,b_value,b_loss,b_left,b_right = 999,999,999,None,None
    while len(features) < n_features:#n_features代表要选取的特征数目
        index = randrange(len(dataSet[0])-1)#从所有的特征中随机选取一个
        if index not in features:
            features.append(index)

    for index in features:
        for row in dataSet:
            left,right = data_split(dataSet,index,row[index])
            loss = split_loss(left,right,class_values)
            if loss < b_loss:
                b_index, b_value, b_loss, b_left, b_right = \
                index,row[index],loss,left,right

    return {"index":b_index,"value":b_value,"left":b_left,"right":b_right}

#决定输出标签
def decide_label(data):
    output=[row[-1] for row in data]
    return max(set(output),key=output.count)

#子分割，不断地构建叶节点的过程
def sub_split(root,n_features,max_depth,min_size,depth):
    left=root['left']
    #print left
    right=root['right']
    del(root['left'])
    del(root['right'])
    #print depth
    if not left or not right:
        root['left']=root['right']=decide_label(left+right)
        #print 'testing'
        return
    if depth > max_depth:#如果决策树的深度大于max_depth时还没有分类完，就强制停止，防止过拟合
        root['left']=decide_label(left)
        root['right']=decide_label(right)
        return
    if len(left) < min_size:#如果分类的列表大小小于min_size，则强制停止
        root['left']=decide_label(left)
    else:
        root['left'] = get_best_split(left,n_features)#再次进行决策树的递归
        #print 'testing_left'
        sub_split(root['left'],n_features,max_depth,min_size,depth+1)
    if len(right) < min_size:
        root['right']=decide_label(right)
    else:
        root['right'] = get_best_split(right,n_features)
        #print 'testing_right'
        sub_split(root['right'],n_features,max_depth,min_size,depth+1)

#4.构造决策树
def build_tree(dataSet,n_features,max_depth,min_size):
    root = get_best_split(dataSet,n_features)
    sub_split(root,n_features,max_depth,min_size,1)
    return root


# 预测测试集结果
def predict(tree, row):
    predictions = []
    if row[tree['index']] < tree['value']:
        if isinstance(tree['left'], dict):
            return predict(tree['left'], row)
        else:
            return tree['left']
    else:
        if isinstance(tree['right'], dict):
            return predict(tree['right'], row)
        else:
            return tree['right']
            # predictions=set(predictions)


def bagging_predict(trees,row):
    predictions=[predict(tree,row) for tree in trees]
    return max(set(predictions),key=predictions.count)

#5.创建随机森林
def random_forest(train,test,ratio,n_feature,max_depth,min_size,n_trees):
    trees = []
    for i in range(n_trees):
        subTrain = get_subsample(train,ratio)
        tree = build_tree(subTrain,n_feature,max_depth,min_size)
        trees.append(tree)
    predict_values = [bagging_predict(trees,row) for row in test]
    return predict_values

#计算准确率
def accuracy(predict_values,actual):
    correct=0
    for i in range(len(actual)):
        if actual[i]==predict_values[i]:
            correct+=1
    return correct/float(len(actual))


random.seed(1)
dataSet=loadCSV('D:\python_machine-learning\Code\Machine-learning\Random-Forest\RandomForest/sonar-all-data.csv')
column_to_float(dataSet)
n_folds=5
max_depth=15
min_size=1
ratio=1.0
#n_features=sqrt(len(dataSet)-1)
n_features=15
n_trees=10
folds=splitDatset(dataSet,n_folds)
scores=[]
for fold in folds:
    train_set=folds[:]  #此处不能简单地用train_set=folds，这样用属于引用,那么当train_set的值改变的时候，folds的值也会改变，所以要用复制的形式。（L[:]）能够复制序列，D.copy() 能够复制字典，list能够生成拷贝 list(L)
    train_set.remove(fold)
    #print len(folds)
    train_set=sum(train_set,[])  #将多个fold列表组合成一个train_set列表
    #print len(train_set)
    test_set=[]
    for row in fold:
        row_copy=list(row)
        row_copy[-1]=None
        test_set.append(row_copy)
    #for row in test_set:
       # print row[-1]
    actual=[row[-1] for row in fold]
    predict_values=random_forest(train_set,test_set,ratio,n_features,max_depth,min_size,n_trees)
    accur=accuracy(predict_values,actual)
    scores.append(accur)
print ('Trees is %d'% n_trees)
print ('scores:%s'% scores)
print ('mean score:%s'% (sum(scores)/float(len(scores))))

