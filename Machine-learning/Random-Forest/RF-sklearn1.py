from sklearn.ensemble import RandomForestClassifier as RFC
import csv
def loadCSV(filename):
    dataSet = []
    with open(filename,'r') as file:
        csvReader = csv.reader(file)
        for line in csvReader:

            dataSet.append(line)
    return dataSet

#除了判别列，其他列均转换为float类型
def column_to_float(dataSet):
    featLen = len(dataSet[0])-1#判断需要有多少列需要转化
    for data in dataSet:
        for column in range(featLen):
            data[column] = float(data[column].strip())#strip()返回移除字符串头尾指定的字符生成的新字符串。


def loadData(dataSet):
    data = []
    label = []
    for line in dataSet:
        data.append(line[:-1])
        label.append(line[-1])
    return data,label




dataSet=loadCSV('D:\python_machine-learning\Code\Machine-learning\Random-Forest\RandomForest/sonar-all-data.csv')
column_to_float(dataSet)
data,label = loadData(dataSet)
clf = RFC(n_estimators=10)
clf.fit(data,label)
print(clf.score(data,label))

