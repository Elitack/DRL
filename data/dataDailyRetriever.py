# pylint: disable=I0011,C0103,C0326,C0301, W0401,W0614
from cassandra.cluster import Cluster
from WindPy import *
import time
import datetime
import os
import math
import logWriter

def dailyRetrieve(startTime, endTime, logDir,
fields1 = ['trade_status','close', 'mfd_buyamt_d', 'mfd_sellamt_d'],
option1 = "unit=1;traderType=1;Period=D;Fill=Previous;PriceAdj=B", multi_mfd = True):
    # cassandra connect
    cluster = Cluster(['192.168.1.111'])
    session = cluster.connect('factors') # factors: factors_month

    # 启动Wind API
    w.start()

    # 获取可交易日
    times = w.tdays(startTime, endTime, "Period=D").Times
    timeList = []
    for i in range(len(times)):
        row = str(times[i])
        row = row[:row.find(' ')]
        timeList.append(row)
    print(timeList)
    print("--- Total days: ", len(timeList))

    rows = session.execute('''SELECT stock, ipo_date FROM stock_info WHERE trade_status = '1' ALLOW FILTERING ''')
    validStocks = {}
    validStockCode = []
    for row in rows:
        validStocks[row.stock] = row.ipo_date
        validStockCode.append(row.stock)

    validN = len(validStocks)
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , " valid stocks' number: ", validN)

    ## 拉取机构/大户/散户买入卖出因子，分阶段拉取，拉完异步存DB
    if multi_mfd == True:
        # columns = fields1 + ['mfd_buyamt_d2', 'mfd_sellamt_d2','mfd_buyamt_d4', 'mfd_sellamt_d4']
        columns = fields1 + ['mfd_buyamt_d1', 'mfd_sellamt_d1','mfd_buyamt_d4', 'mfd_sellamt_d4']
    else:
        columns = fields1
    print(columns)
    # 拉取交易状态便于之后数据过滤
    hasTradeStatus = False
    if len(fields1) >= 1 and fields1[0] == 'trade_status':
        hasTradeStatus = True

    dataList = [] #创建数组
    cnt = 0   #当前拉取了多少支股票
    index = 0 #上一次dump的位置，主要目的是通过此索引找到该股票代码
    CHUNK_SIZE = 30 #每一次异步dump的股票个数

    preparedStmt = session.prepare('''INSERT INTO factors_day(stock, factor, time, value) VALUES (?,?,?,?)''')
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , " ------ Starting to insert to DB")

    ## 遍历所有股票
    for stock,ipo_date in validStocks.items():
        # 日数据中无需ROA，只拉取IPO之后的数据减少数据传输
        # start = startTime if startTime > ipo_date.date() else ipo_date.date()
        start = startTime
        wsd = w.wsd(stock, fields1, start, endTime, option1)
        if wsd.ErrorCode != 0:
            print("--------------------- ERROR IN WIND ------------\r\n ErrorCode：", wsd.ErrorCode, " Stock: ",stock)
        wsd_data = wsd.Data
        if multi_mfd == True:
            # 同一个变量，参数不一样，需要分成几次拉取
            # fields2 = ['mfd_buyamt_d', 'mfd_sellamt_d']
            # 只需要散户买入
            fields2 = ['mfd_buyamt_d', 'mfd_sellamt_d']
            option2 = "unit=1;traderType=1;Period=D;Fill=Previous;PriceAdj=B"
            wsd_data = wsd_data + w.wsd(stock, fields2, start, endTime, option2).Data
            option3 = "unit=1;traderType=4;Period=D;Fill=Previous;PriceAdj=B"
            wsd_data = wsd_data + w.wsd(stock, fields2, start, endTime, option3).Data

        dataList.append(wsd_data)
        cnt += 1
        #阶段性异步导出 dump data asynchronously, 30 stocks / round
        if cnt % CHUNK_SIZE == 0:
            filename = logDir+"\\"+str(startTime)+"_"+str(endTime)+"_"+str(index)+".sql"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as file:
                for s in range(index, cnt):
                    for i in range(len(columns)):
                        for j in range(len(dataList[s - index][i])):
                            #print (validStocks[s],columns[i],timeList[j],dataList[s - index][i][j])
                            try:
                                value = dataList[s - index][i][j]
                                if hasTradeStatus == True and i == 0:
                                    # 交易 状态作为一个因子
                                    if  value is not None and value == "交易":
                                        value = 1
                                    else:
                                        value = 0
                                elif value is not None:
                                    value = float(value)
                                else:
                                    value = float('nan')
                            except (ValueError, TypeError, KeyError) as e:
                                value = float('nan')
                                print ("--Log ValueError in ", validStockCode[s],"\t",columns[i],"\t",str(timeList[j]),"\t",str(value))
                                print (e)
                                print ("--------------------------------------------------------------------------")
                            except IndexError as e:
                                print ("--------------------------------------------------------------------------")
                                print("len s: %d, len i: %d, len j: %d ~ " %(cnt, len(columns),len(timeList)), (s-index,i,j))
                                print(e)
                            # session.execute(preparedStmt, (validStockCode[s],columns[i],timeList[j], value))
                            # 写入文件做log， 之后用程序异步执行插入
                            if value is None or math.isnan(value) is True :
                                value = 0
                            file.write("INSERT INTO factors_day(stock, factor, time, value) VALUES (\'"+validStockCode[s]+"\', \'"+columns[i]+"\',\'"+str(timeList[j])+"\',"+str(value)+" );\n")
            #记录上一次导出数据位置，清空buffer
            index = cnt
            dataList = []
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,'------ Dump NO.%d end at stock %s \n' % (cnt, stock))

    print ("---- Last chunk size: ", len(dataList))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,'---------------- Pulling finished!\n')

    # 最后的剩余数据插入cassandra
    filename = logDir+"\\"+str(startTime)+"_"+str(endTime)+"_"+str(index)+".sql"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        for s in range(index, cnt):
            for i in range(len(columns)):
                for j in range(len(dataList[s - index][i])):
                    #print (validStocks[s],columns[i],timeList[j],dataList[s - index][i][j])
                    try:
                        value = dataList[s - index][i][j]
                        if hasTradeStatus == True and i == 0:
                            if  value is not None and value == "交易":
                                value = 1
                            else:
                                value = 0
                        elif value is not None:
                            value = float(value)
                        else:
                            value = float('nan')
                    except (ValueError, TypeError, KeyError) as e:
                        value = float('nan')
                        print ("--Log ValueError in ", validStockCode[s],"\t",columns[i],"\t",str(timeList[j]),"\t",str(value))
                        print (e)
                        print ("--------------------------------------------------------------------------")
                    except IndexError as e:
                        print ("--------------------------------------------------------------------------")
                        print("len s: %d, len i: %d, len j: %d ~ " %(cnt, len(columns),len(timeList)), (s-index,i,j))
                        print(e)
                    # session.execute(preparedStmt, (validStockCode[s],columns[i],timeList[j], value))
                    if value is None or math.isnan(value) is True :
                        value = 0
                    file.write("INSERT INTO factors_day(stock, factor, time, value) VALUES (\'"+validStockCode[s]+"\', \'"+columns[i]+"\',\'"+str(timeList[j])+"\',"+str(value)+" );\n")

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '---------------- Persistion finished!\n')

    ############## Dump Log file to cassandra ###############
    logWriter.persist(logDir, session)

    #result testing
    print("---------- Inserstion Testing: ")
    rows = session.execute("select * from factors_day where stock='000852.SZ'  and time > '2017-04-24' ALLOW FILTERING;")
    for row in rows:
        print(row.stock,row.factor,row.time,row.value)

    # close connection with cassandra
    cluster.shutdown()

# dailyRetrieve(datetime.date(2017,4,24), datetime.date(2017,4,24), multi_mfd = False)
# dailyRetrieve(datetime.date(2017,4,7), datetime.datetime.today(), fields1=['close'], multi_mfd = False)
# dailyRetrieve(datetime.date(2017,4,27), datetime.datetime.today().date(),"G:\\log\\daily_mfd_buyamt_d\\4-28", fields1 = ['trade_status','mfd_buyamt_d'],multi_mfd = False)
dailyRetrieve(datetime.date(2012,1,1), datetime.datetime.today().date(),"G:\\log\\daily_mfd_buy_sell_amt_12-17_5", fields1 = ['trade_status'],multi_mfd = True)
