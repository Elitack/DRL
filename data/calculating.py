# pylint: disable=I0011,C0103,C0326,C0301, W0401,W0614
from cassandra.cluster import Cluster
from cassandra.util import Date
from datetime import timedelta
import time
import datetime
import math
## calculating ROA growth
# 1. get all the time series
# 2. mark the last date of the endYear
# 3. calulate ROA growth
# 4. insert into DB
## ROA: 今年此月的ROA / 去年的此月的ROA

def calculate_ROA_growth(beginDate, endDate, factor_table = "factors_month"):
    #cassandra connection
    cluster = Cluster(['192.168.1.111'])
    session = cluster.connect('factors') #connect to the keyspace 'factors'

    # get stocks list with IPO_date
    # rows = session.execute('''SELECT stock, ipo_date FROM stock_info WHERE stock = '603636.SH' and trade_status = '1' ALLOW FILTERING ''')
    rows = session.execute('''SELECT stock, ipo_date FROM stock_info WHERE trade_status = '1' ALLOW FILTERING ''')
    stocks = {}
    for row in rows:
        stocks[row[0]] = row[1]

    # 得到去年的12个月的ROA值，以计算今年的ROA值，用372而不是365是为了留出空余给非交易日
    lastYearOfBeginDate = beginDate - timedelta(days=372)
    selectPreparedStmt = session.prepare("select * from "+factor_table+" where stock = ? and factor = 'roa' and time >= ? and time <= ? ALLOW FILTERING")
    roaFillPrevStmt = session.prepare("INSERT INTO "+factor_table+" (stock, factor, time, value) VALUES (?,'roa', ?, ?)")
    preparedStmt = session.prepare("INSERT INTO "+factor_table+" (stock, factor, time, value) VALUES (?,'roa_growth', ?, ?)")
    #select all ROA value for each stock
    for stock, ipo_date in stocks.items():
        # 暴力除法，最后过滤
        # isOverlappd = False
        # if lastYearOfBeginDate > ipo_date.date():
        #     begin = lastYearOfBeginDate
        # else:
        #     begin = ipo_date.date()
        #     isOverlappd = True
        # rows = session.execute(selectPreparedStmt, (stock, str(begin), str(endDate)))
        rows = session.execute(selectPreparedStmt, (stock, str(lastYearOfBeginDate), str(endDate)))

        ## Fill in the NaN value with previous non-NaN value
        ## store historic ROA in tuple array
        roaMap = []
        prev = float('nan')
        for row in rows:
            if math.isnan(row.value) is True and math.isnan(prev) is False:
                session.execute(roaFillPrevStmt,(row.stock, row.time, prev))
                print (" Fill in %s  %s  %d"%(stock, str(row.time), prev))
            else:
                prev = row.value
            roaMap.append((row.time, prev))

        ## calculate growth # ROA_curr / ROA_prev[-12month]
        index = 0
        length = len(roaMap)
        # 【修正】暴力解法，不用判断IPO，因为IPO之前可能还有ROA
        # 1. IPO没越界, DB中前12个月无数据
        # 2. IPO越界，但小于BeginDate
        # 以第一个不小于BeginDate的交易日作为起点
        # if isOverlappd == False or (isOverlappd == True and ipo_date.date() < beginDate):
        #     while index < length and roaMap[index][0].date() < beginDate:
        #         index += 1
        # if index < length:
        #     print ("%s Begin: %s , IPO: %s index: %d , len: %d" % (stock, str(roaMap[index][0].date()), str(ipo_date.date()),index, length))

        # calculate
        roa_growth = {}
        for i in range(index, length):
            # 前面没有数据
            if i < 12:
                roa_growth[roaMap[i][0].date()] = float('nan')
            else:
                prevRoa = roaMap[i-12][1]
                roa_growth[roaMap[i][0].date()] = roaMap[i][1] / prevRoa if prevRoa != 0 and math.isnan(prevRoa) == False else float('nan')

        # insert to DB
        for item in roa_growth.items():
            session.execute_async(preparedStmt, (stock, item[0], item[1]))
            # print(item[0].strftime("%Y-%m-%d"), item[1])

        print (stock + " roa_growth calculation finished")

## Yield = Close(next_month_end) / Close(next_month_start)
# eg. Yield(03-31) = Close(04-30) / Close(04-01)
def calculate_Yield(beginDate, endDate, calc_table = "factors_day", store_table = "factors_month", TYPE="D"):
    # cassandra connection
    cluster = Cluster(['192.168.1.111'])
    session = cluster.connect('factors') #connect to the keyspace 'factors'

    # month begin & end time list, 1 month after, 37 reserved for error buffer
    nextMonth = endDate + timedelta(days=37)
    firstDay = datetime.date(beginDate.year, beginDate.month,1)
    sql="select * from transaction_time where type= '"+TYPE+ "' and time >= '"+ str(firstDay) +"' and time <= '" + str(nextMonth)+"'"
    rows = session.execute(sql)
    dateList = []
    prevMonth = beginDate.month
    currMonth = prevMonth
    prevDay = None
    currDay = None
    cnt = 0
    # 筛选出月初和月末的交易日日期
    for row in rows:
        prevDay = currDay
        prevMonth = currMonth
        # update
        currDay = row.time.date()
        if cnt == 0:
            dateList.append(currDay)
        currMonth = currDay.month
        #print('currDay: %s currentMonth: %s' % (currDay, currMonth))
        # month change, add previous month end & this month start
        if currMonth != prevMonth:
            if prevDay is not None:
                dateList.append(prevDay)
            dateList.append(currDay)
        cnt += 1
    # omit 1st one when it's the end of month
    # add the last day 月末
    dateList.append(currDay)
    print(" Size of dateList: ",len(dateList))

    # if dateList[1].month != beginDate.month:
    #     dateList = dateList[1:]
    # print(dateList)
    # make it even, 凑齐月初,月末
    if len(dateList) % 2 != 0:
        dateList = dateList[:-1]
    print(dateList)
    if(len(dateList) == 0):
        print("Length 0 DateList, EXIT ")
        exit()
    insertPreparedStmt = session.prepare("INSERT INTO "+store_table+" (stock, factor, time, value) VALUES (?,'Yield', ?, ?)")

    # get stocks list with IPO_date
    rows = session.execute('''SELECT stock, ipo_date FROM stock_info WHERE trade_status = '1' ALLOW FILTERING ''')
    # rows = session.execute('''SELECT stock, ipo_date FROM stock_info WHERE stock = '603636.SH' and trade_status = '1' ALLOW FILTERING ''')
    stocks = {}
    for row in rows:
        stocks[row[0]] = row[1]
    #select all daily close price value for each stock
    #for stock in ["000852.SZ","603788.SH","603990.SH","603991.SH","603993.SH"]:
    for stock, ipo_date in stocks.items():
        sql = "select time, value from " + calc_table + " where stock = '"+stock+"' and factor = 'close' and time in ("
        for day in dateList:
            # if day > ipo_date.date():        # delete invalid date [UPDATE: final filter will do this]
            sql += "'"+str(day)+"',"
        sql = sql[:-1] + ");"               # omit the extra comma
        rows = session.execute(sql)
        # print(sql)
        ## calculating close Yield
        # divid the first day's close price by the end day in the month
        prev = 1.0     # previous Yield value
        yield_map = {}
        cnt = 0
        size = len(dateList)
        prevTime = beginDate
        for row in rows:
            # end of month, store the quotient
            if cnt % 2 > 0:
                if cnt > 1:
                    value = row.value
                    if math.isnan(value):
                        value = 0 # last data if not available
                    yield_map[prevTime] = float(value) / float(prev)
                prevTime = row.time
            # in case divided by 0
            elif row.value == 0:
                prev = float('nan')
            else:
                prev = row.value
            #print ("cnt " + str(cnt)+" K: ", row[0]," V: ", row[1])
            cnt = cnt + 1

        # last Value is always 0, would be updated when new data comes
        yield_map[dateList[-1]] = 0
        # insert to DB
        for item in yield_map.items():
            session.execute_async(insertPreparedStmt, (stock, item[0], item[1]))
            # print(str(item[0]), item[1])
        # print (stock + " Yield calculation finished",cnt)
    print(str(len(stocks))," Stock's Yield Calculation Complete ")
    cluster.shutdown()

# 计算动量模块单独抽取出来，默认为1个月的动量，因为之后可能要计算两个月，三个月的动量
# mmt = Close(this month) / Close(last month)
def calculate_mmt(beginDate, endDate, factor_table = "factors_month", gap = 1):
    cluster = Cluster(['192.168.1.111'])
    session = cluster.connect('factors')

    # tradable stocks' collection
    rows = session.execute('''SELECT stock, ipo_date FROM stock_info WHERE trade_status = '1' ALLOW FILTERING ''')
    stocks = {}
    for row in rows:
        stocks[row[0]] = row[1]

    preparedStmt = session.prepare("INSERT INTO "+factor_table+" (stock, factor, time, value) VALUES (?,'mmt', ?, ?)")
    selectPreparedStmt = session.prepare("select time, value from "+factor_table+" where stock = ? and factor = 'close' and time >= ? and time <= ? ALLOW FILTERING")
    #select all close value for each stock every month
    for stock, ipo_date in stocks.items():
        # 1 month ahead
        lastMonth = beginDate - timedelta(days=37)
        # begin = beginDate if beginDate > ipo_date.date() else ipo_date.date() ## final filter will do the duty
        rows = session.execute(selectPreparedStmt, (stock, str(lastMonth), str(endDate)))
        ## calculating mmt
        cnt = 0
        curr = 0
        prev = 0            # previous close value
        mmt = 0             # Close_curr / Close_prev
        mmt_dic = {}
        for row in rows:
            if cnt == 0:
                curr = row.value
                cnt += 1
                continue
            prev = curr
            curr = row.value
            # if math.isnan(curr):
            #     curr = 0
            # in case divided by 0
            mmt = curr / prev if prev != 0 else float('nan')
            mmt_dic[row.time] = mmt
            cnt += 1
        print (stock + " mmt calculation finished", str(len(mmt_dic)), cnt)
        # insert to DB
        for item in mmt_dic.items():
            session.execute_async(preparedStmt, (stock, item[0], item[1]))
            # print(item[0].date().strftime("%Y-%m-%d"), item[1])

    print(str(len(stocks))," Stock's Momentum Calculation Complete ",cnt-1," days")

##一个月一个月地对每个mfd_buy_sell_d求和
# 注意：先变成万元使数字变小，减少运算复杂度（不会溢出，因为浮点数在计算机内用科学计数法存储double:-2^1024 ~ 2^1024

def calculate_mfd_sum(beginDate, endDate, factors=['mfd_buyamt_d', 'mfd_sellamt_d','mfd_buyamt_d2', 'mfd_sellamt_d2','mfd_buyamt_d4', 'mfd_sellamt_d4']):
    cluster = Cluster(['192.168.1.111'])
    session = cluster.connect('factors')

    # IPO距今至少三个月的合理股票
    rows = session.execute('''SELECT stock FROM stock_info WHERE trade_status = '1' ALLOW FILTERING ''')
    stocks = []
    for row in rows:
        stocks.append(row[0])

    # 所有月末时间
    rows = session.execute("select time from transaction_time WHERE type='M' and time >= %s and time<= %s ",(beginDate,endDate))
    days = []
    for row in rows:
        days.append(row[0])

    # 上个月的月末作为起始点（不包括exclusive），本月的月末作为终点（包括inclusive）
    selectPreparedStmt = session.prepare("select value from factors_day where stock = ? and factor = ? and time > ? and time <= ? ALLOW FILTERING")
    insertPreparedStmt = session.prepare("INSERT INTO factors_month (stock, factor, time, value) VALUES (?, ?, ?, ?)")

    for stock in stocks:
        for factor in factors:
            prevDay = beginDate
            for day in days:
                mfd_sum = 0.0
                rows = session.execute(selectPreparedStmt, (stock, factor, prevDay, day))
                for row in rows:
                    if row.value is not None:
                        value  = row.value / 10000.0
                        mfd_sum += value
                # print(prevDay, " -- ", day, stock, factor, day, mfd_sum)
                prevDay = day
                # insert into db
                session.execute(insertPreparedStmt, (stock, factor, day, mfd_sum))
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '---------------- Calculate %s %s complete!'%(stock, str(factors)))
################################
#### Invoke Function  ##########
# calculate_ROA(datetime.date(2009,1,1), datetime.date(2017,4,1), "factors_month")
calculate_ROA_growth(datetime.date(2015,1,1), datetime.date(2017,4,30), "factors_month")
# calculate_Yield(datetime.date(2009,1,1), datetime.datetime.today().date())
# calculate_mmt(datetime.date(2009,1,1), datetime.datetime.today().date())
# calculate_mmt(datetime.date(2017,4,1), datetime.date(2017,4,30))
# calculate_Yield(datetime.date(2017,3,31), datetime.date(2017,4,30))
# calculate_mfd_sum(datetime.date(2017,4,1), datetime.date(2017,4,30),factors=['mfd_buyamt_d','mfd_buyamt_d4'])