# pylint: disable=I0011,C0103,C0326,C0301, W0401,W0614
from WindPy import *
import datetime
import time
from cassandra.cluster import Cluster

### 获取基本股票信息
## 可交易日/周/月
## 所有股票名/IPO日期/可交易状态(待修正，每次查询A股都要判断是否处于可交易状态，因此严格来说每次数据获取和导出时都要加判定)

def updateAStocks(day, extraIndex = []):
    # 启动Wind API
    w.start()
    #取全部 A 股股票代码、名称信息(不写field，默认为wind_code & sec_name & date)
    stocks = w.wset("SectorConstituent","date="+str(day)+";sectorid=a001010100000000;field=wind_code,sec_name")
    data = stocks.Data
    if(stocks.ErrorCode !=0 ):
        print("WSET ERROR CODE : ", stocks.ErrorCode)
        exit()
    print("---- Total stock number: ", len(data[0]))

    # indexInfo = []
    # for index in extraIndex:
    #     d = w.wsd(index, "windcode,sec_name", "2017-04-06", "2017-04-06", "Period=Y").Data
    #     indexInfo.append((d[0][0], d[1][0]))
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ," TOTAL A STOCK'S SIZE: ", size)
    # print ("index info: "+indexInfo)

    cluster = Cluster(['192.168.1.111'])
    session = cluster.connect('factors') #connect to the keyspace 'factors'
    # preparedStmt = session.prepare('''INSERT INTO stock_info(stock, sec_name) VALUES (?,?)''')
    # for i in range(size):
    #     session.execute(preparedStmt,(data[0][i],data[1][i]))
    # print ("Updating stocks of A share's name complete!")
    # for i in range(len(indexInfo)):
    #     session.execute(preparedStmt,(indexInfo[i][0],indexInfo[i][1]))
    # print ("Updating Stock Index's name complete!")

    # 更新IPO_DATE & TRADE_STATUS
    # stock status update statement
    updateStmt = session.prepare('''INSERT INTO stock_info(stock, ipo_date, trade_status) VALUES (?,?,?)''')
    validStocks =[]
    # 判断数据有效性
    #for stock in ["000852.SZ","603788.SH","603987.SH","603988.SH","603989.SH","603990.SH","603991.SH","603993.SH"]:
    # for stock in ["000852.SZ","603788.SH","603990.SH","603991.SH","603993.SH"]:
    for stock in data[0]:
        ipo_status = w.wsd(stock, "ipo_date, trade_status", day)
        if(ipo_status.ErrorCode !=0 ):
            print("WSD ERROR CODE : ", ipo_status.ErrorCode)
            exit()
        try:
            days = (datetime.datetime.today() - ipo_status.Data[0][0]).days
            # if  days > 90 and ipo_status.Data[1][0] == "交易":
            if  days > 92:
                validStocks.append(stock)
                session.execute(updateStmt, (stock, ipo_status.Data[0][0], '1')) # status 1 : "交易"
            else:
                # set status 0
                session.execute(updateStmt, (stock, ipo_status.Data[0][0], '0')) # status 0 : "不可交易"
                print (" Set invalid data: ", stock, str(ipo_status.Data[0][0]))

        except TypeError:
            print (" -- Log TypeError at Stock: ", stock, " :\t", str(ipo_status.Data[0][0]))
    validN = len(validStocks)
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , " valid stocks' number: ", validN)

    # Set index's status as '2', different from others
    for index in extraIndex:
        session.execute('''INSERT INTO stock_info(stock, trade_status) VALUES (%s,%s)''', (index, '2')) # status 2 : "指数"
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , " Updating Indexes Complete! ")
#取沪深 300 指数中股票代码和权重
#stocks = w.wset("IndexConstituent","date=20130608;windcode=000300.SH;field=wind_code,i_weight")

# # Testing
# print ("testing: select * from transaction_time where type='month' and time > '2016-03-02' ALLOW FILTERING;")
# rows = session.execute("select * from transaction_time where type='month' and time > '2016-03-02' ALLOW FILTERING;")
# for row in rows:
#     print(row.time)

#从周因子表中获取股票600000.SH在2017-03-02至今的所有因子
# rows = session.execute("select * from factors_week where stock='600000.SH' and time > '2017-03-02' ALLOW FILTERING;")
# for row in rows:
#     print(row.stock,row.factor,row.time,row.value)

## Testing： get all stocks 
# rows = session.execute('''select stock from stock_info''')
# stocks = []
# for row in rows:
#     stocks.append(row[0])
# print (stocks)

## insert transaction day
def updateTransactionTime(startTime, endTime = datetime.datetime.today(),TYPE='D'):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),"Updating Transaction Time in TYPE: ", TYPE)
    # 启动Wind API
    w.start()
    times = w.tdays(startTime, endTime, "Period="+TYPE).Times
    timeList = []
    for i in range(len(times)):
        row = str(times[i])
        row = row[:row.find(' ')]
        timeList.append(row)
    print(timeList)

    #cassandra connect  to the keyspace 'factors'
    cluster = Cluster(['192.168.1.111'])
    session = cluster.connect('factors')
    preparedStmt = session.prepare('''INSERT INTO transaction_time(type, time) VALUES (?,?)''')
    for date in timeList:
        session.execute(preparedStmt, (TYPE, date))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())," Updating Complete!")


def updateStatus(startTime, endTime):
    # 启动Wind API
    w.start()
    # #取全部 A 股股票代码、名称信息(不写field，默认为wind_code & sec_name & date)
    # stocks = w.wset("SectorConstituent",u"sector=全部A股;field=wind_code,sec_name")
    # data = stocks.Data
   
    cluster = Cluster(['192.168.1.111'])
    session = cluster.connect('factors') #connect to the keyspace 'factors'
    
    # select all A share
    rows = session.execute('''SELECT stock FROM stock_info ALLOW FILTERING ''')
    stocks = []
    for row in rows:
        stocks.append(row[0])

    # 更新IPO_DATE & TRADE_STATUS
    # stock status update statement
    updateStmt = session.prepare('''INSERT INTO stock_info(stock, ipo_date, trade_status) VALUES (?,?,?)''')
    
    
    validStocks =[]
    ipo_valid =[]
    trade_valid =[]
    # 判断数据有效性
    #for stock in ["000852.SZ","603788.SH","603987.SH","603988.SH","603989.SH","603990.SH","603991.SH","603993.SH"]:
    # for stock in ["000852.SZ","603788.SH","603990.SH","603991.SH","603993.SH"]:
    for stock in stocks:
        ipo_status = w.wsd(stock, "ipo_date, trade_status", startTime)
        try:
            days = (datetime.datetime.today() - ipo_status.Data[0][0]).days
            if  days > 90:
                ipo_valid.append(stock)
                if ipo_status.Data[1][0] == "交易":
                    trade_valid.append(stock)
                    validStocks.append(stock)
                # session.execute_async(updateStmt, (stock, ipo_status.Data[0][0], '1')) # status 1 : "交易"
            elif ipo_status.Data[1][0] == "交易":
                trade_valid.append(stock)
                # set status 0
                # session.execute_async(updateStmt, (stock, ipo_status.Data[0][0], '0')) # status 0 : "不可交易"
            else:
                print (" Set invalid data (IPO < 90 && 不可交易): ", stock, str(ipo_status.Data[0][0]))
        except TypeError:
            print (" -- Log TypeError at Stock: ", stock, " :\t", str(ipo_status.Data[0][0]))
    validN = len(validStocks)
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , " ipo > 90 &&  可交易 stocks' number: ", validN)
    print (" ipo > 90 stocks' number: ", len(ipo_valid))
    print (" 可交易 stocks' number: ", len(trade_valid))


#########################################################
## Updating Available A share Stock & transaction_time ##
#########################################################
# transaction day
# updateTransactionTime('2017-04-01')
# updateTransactionTime('2017-04-01', TYPE='M')
# update stocks
updateAStocks(datetime.date(2017,4,28), extraIndex=["000001.SH","399001.SZ",'399006.SZ','000300.SH','000016.SH','000905.SH'])
# updateStatus("2015-8-31","2015-8-31")
