# pylint: disable=I0011,C0103,C0326,C0301, W0401,W0614

from datetime import datetime
from WindPy import *
import time
# 启动Wind API
w.start()

# 获取每个时间点所有的A股 IPO > 3个月的股票 目前共[3150]支，有[3014]支符合要求
stocks = w.wset("SectorConstituent", u"sector=全部A股;field=wind_code")
validStocks =[]
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "Total A stocks number: ", len(stocks.Data[0]))

#for stock in ["000852.SZ","603788.SH","603987.SH","603988.SH","603989.SH","603990.SH","603991.SH","603993.SH"]:
#for stock in ["000852.SZ","603788.SH","603990.SH","603991.SH","603993.SH"]:
for stock in stocks.Data[0]:
    ipo_status = w.wsd(stock, "ipo_date, trade_status", datetime.today())
    #print (ipo_status)
    if (datetime.today() - ipo_status.Data[0][0]).days > 90 and ipo_status.Data[1][0] == "交易":
        validStocks.append(stock)

validN = len(validStocks)
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , " valid stocks' number: ", validN)

# 获取周数据（有些字段参数不同，需分成若干次拉取）
#fields1 = ['windcode','sec_name','ipo_date','close', 'mkt_cap_float', 'mfd_buyamt_d', 'mfd_sellamt_d', 'roa', 'pe', 'pb']
#收盘价（向后复权）；自由流通市值；流入额；流出额
#fields1 = ['close','mkt_freeshares','mfd_buyamt_d', 'mfd_sellamt_d', 'roa', 'pe', 'pb']
# 定义某些指标需要的参数，收盘价向后复权， 流入流出为机构
option1 = "ruleType=8;unit=1;traderType=1;Period=W;Fill=Previous;PriceAdj=B"

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , 'Pulling Started...\n --------------------------')

startTime = '2013-01-01'
endTime = datetime.today() # 可缺省，默认今天

times = w.tdays(startTime, endTime, "Period=W").Times
timeList = []
for i in range(len(times)):
    row = str(times[i])
    row = row[:row.find(' ')]
    timeList.append(row)
print(timeList)

#将所有的股票的数据拉取填充数据，然后存入Cassandra。优化方案，每次只拉小部分股票，直接BATCH操作存入cassandra, 迭代多次
# cassandra connection
from cassandra.cluster import Cluster

cluster = Cluster(['192.168.1.111'])
session = cluster.connect('factors') #connect to the keyspace 'factors'
preparedStmt = session.prepare('''INSERT INTO factors_week(stock, factor, time, value) VALUES (?,?,?,?)''')

dataList = [] #创建数组
cnt = 0
index = 0
for stock in validStocks:
    wsd_data = w.wsd(stock, "mkt_freeshares", startTime, endTime, option1).Data
    #print (wsd_data)
    #fields2 = ['mfd_buyamt_d', 'mfd_sellamt_d']
    #option2 = "unit=1;traderType=2;Period=W;Fill=Previous;PriceAdj=B"
    #wsd_data = wsd_data + w.wsd(stock, fields2, startTime, endTime, option2).Data
    #print (wsd_data)
    #option3 = "unit=1;traderType=4;Period=W;Fill=Previous;PriceAdj=B"
    #wsd_data = wsd_data + w.wsd(stock, fields2, startTime, endTime, option3).Data
    #print (wsd_data)
    dataList.append(wsd_data)
    cnt += 1
    #dump data asynchronously, 300 stocks / round
    if cnt % 300 == 0:
        for s in range(index, cnt):
            for j in range(len(timeList)):
                #print (validStocks[s],'mkt_freeshares',timeList[j],str(dataList[s - index][0][j]))
                session.execute_async(preparedStmt, (validStocks[s],'mkt_freeshares',timeList[j],str(dataList[s - index][0][j])))
        index = cnt
        dataList = []
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,'---------------- Dump NO.%d end at %s \n' % (cnt, stock))

print ("---- Last chunk size: ", len(dataList))
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,'---------------- Pulling finished!\n')

# 插入cassandra
for s in range(index, cnt):
    for j in range(len(timeList)):
        #print (validStocks[s],' mkt_freeshares ',timeList[j],str(dataList[s - index][0][j]))
        session.execute_async(preparedStmt, (validStocks[s],'mkt_freeshares',timeList[j],str(dataList[s - index][0][j])))

# columns = ['close','mkt_cap_float','mfd_buyamt_d1', 'mfd_sellamt_d1', 'roa', 'pe', 'pb','mfd_buyamt_d2', 'mfd_sellamt_d2','mfd_buyamt_d4', 'mfd_sellamt_d4']
# for s in range(len(validStocks)):
#     for i in range(len(columns)):
#         for j in range(len(timeList)):
#             #execute insertion asynchronously
#             session.execute_async(preparedStmt, (validStocks[s],columns[i],timeList[j],str(dataList[s][i][j])))
            
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '---------------- Persistion finished!\n')

#result testing
print("---------- Inserstion Testing: ")
rows = session.execute("select * from factors_week where stock='000852.SZ' and factor = 'mkt_freeshares' and time > '2017-03-02'")
for row in rows:
    print(row.stock,row.factor,row.time,row.value)

# #获取数据，行列反过来的
# rows = len(wsd_data.Data[0])
# cols = len(wsd_data.Data)
# print ("rows: %d ; columns: %d" % (rows, cols))

# col_name = '\t'
# for x in wsd_data.Fields:
#     col_name = col_name + str(x) + '\t'
# print('\n'+col_name)

# #print (wsd_data.Data)

# for y in range(rows):
#     row = str(wsd_data.Times[y])
#     row = row[:row.find(' ')]
#     for x in range(cols):
#         if x == 2:
#              time = str(wsd_data.Data[x][y])
#              time = time[:time.find(' ')]
#              row = row + '\t' + time
#         elif wsd_data.Data[x][y] is not None:
#             row = row + '\t' + str(wsd_data.Data[x][y])
#     print(row)
#     row = ""
# print('\n--------------- Done ----------------')

# if wsd_data.ErrorCode != 0:
#     print('ErrorCode: ' + str(wsd_data.ErrorCode))

# 生成存入Cassandra的代码，异步执行数据存储，排名计算及定期更新

