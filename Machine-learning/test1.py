import urllib.request
import xlrd
point = []
allMsg1 = []
allMsg = []
data = xlrd.open_workbook('D:/云网火点20180331am.xls')
table = data.sheets()[0]
nrows = table.nrows #行数
ncols = table.ncols #列数
for i in range(1,nrows):
    rowValues= table.row_values(i) #某一行数据
    point.append(rowValues[:2])#着火位置的经纬度
    allMsg1.append(rowValues[2:])


#加载总览的位置信息
position = ''
for i in point:
    i.reverse()#因为API的原因，调换经纬度顺序
    allMsg.append(i)
    singlePos = '{},{}|'.format(i[0],i[1])
    position  = singlePos + position
# print(position)
# print(point)
#
# #着火点总览
# Overview = 'http://uri.amap.com/marker?markers='+ position +'&src=mypage&coordinate=gaode&callnative=0'
# print(Overview)

#各个点地图及信息展示
for i in range(len(allMsg1)):
    allMsg[i].extend(allMsg1[i])
    head = '第{}点:'.format(i + 1)
    pos = '{},{}'.format(allMsg[i][0],allMsg[i][1])
    # name = 'firepoint:{}date:{}time:{}Distance:{}Towernumber:{} '.format(allMsg[i][2],allMsg[i][3],allMsg[i][4],allMsg[i][5],allMsg[i][6])
    name = '第{}点'.format(i + 1)
    url = 'http://restapi.amap.com/v3/staticmap?location='+pos+'&zoom=10&size=800*800&labels='+name+',2,0,16,0xFFFFFF,0x008000:'+pos+'&key=fb9a94655c1ee8b49602d878ca47ebce'
    print(head +url)

