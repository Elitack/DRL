# -*- coding: UTF-8 -*-
# pylint: disable=I0011,C0111, C0103,C0326,C0301, C0304, W0401,W0614
from cassandra.cluster import Cluster
from cassandra.util import Date
import time
import datetime
import math
import numpy as np
#####################################################################################
## Generate 'close' training file within required periods in TXT separated by '\t' ##
#####################################################################################
def exportClose(fileName, startTime, endTime=datetime.datetime.today().date(), table = "factors_day", TYPE='D'):
    if startTime > endTime or len(fileName) == 0:
        return
    cluster = Cluster(['202.120.40.111'])
    session = cluster.connect('factors') #connect to the keyspace 'factors'

    # get valid stocks in A share
    rows = session.execute('''select stock from stock_info''')
    stocks = []
    for row in rows:
        stocks.append(row[0])

    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "Total A stocks number: ", len(stocks))
    #time list
    rows = session.execute('''
        select * from transaction_time 
        where type= %s and time >= %s and time <= %s ALLOW FILTERING;''', [TYPE,startTime, endTime])
    dateList = []
    SQL = "SELECT value FROM "+table+" WHERE stock = ? AND factor = 'close' and time >= '" + datetime.datetime.strftime( startTime,"%Y-%m-%d") +"' and time <= '" + datetime.datetime.strftime(endTime,"%Y-%m-%d")+"'"
    for row in rows:
        #dateList.append(row.time)
        dateList.append(datetime.datetime.strptime(str(row.time), "%Y-%m-%d").strftime('%Y%m%d'))

    # 拉取数据,一次拉一只股票
    dataList = []
    preparedStmt = session.prepare(SQL)
    for stock in stocks:
        rows = session.execute(preparedStmt,(stock,))
        data = []
        for row in rows:
            data.append(row[0])
        dataList.append(data)
    cluster.shutdown()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())," ----- Data are ready!")
    colNum = len(stocks)
    rowNum = len(dateList)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Writing to ',fileName,' ...')
    # 数据写入文件中
    f = open(fileName, "w", encoding='utf-8', newline='\n')
    # f.write(str(colNum))
    # f.write('\t')
    # f.write(str(rowNum))
    # f.write('\n')

    stock_index = np.array(stocks)
    stock_order = np.argsort(stock_index)

    stocks = [stocks[j] for j in stock_order]
    dataList = [dataList[k][:] for k in stock_order]

    f.write('close')
    for stock in stocks:
        f.write('\t'+stock)
    f.write('\n')
    for i in range(rowNum):
        f.write(str(dateList[i]))
        for s in range(colNum):
            try:
                data = dataList[s][i]
                if math.isnan(data):
                    data = 0    # default value
                f.write('\t'+str(data))
            except IndexError:
                print ("End of reading and writing daily close data...")
                f.close()
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Writing to ', fileName, ' complete!')
                return
        f.write('\n')

    f.close()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Writing to ',fileName,' complete!')

##############################################
################# EXAMPLE USAGE ##############
exportClose("E:\\close2.txt",datetime.date(2017,3,1),datetime.date(2017,4,30))