# pylint: disable=I0011,C0111, C0103,C0326,C0301, C0304, W0401,W0614
from cassandra.cluster import Cluster
from cassandra.util import Date
import time
import datetime
import csv
import math
# from dateutil.relativedelta import relativedelta

import os


################################################################################################
## Select all valid stock with required factors identified by stock code & date, saved in CSV ##
################################################################################################
def export(fileName, beginDate, endDate=datetime.datetime.today().date(), factors = [], table = "factors_month"):
    if len(factors) == 0 or beginDate > endDate or len(fileName) == 0:
        return
    # cassandra connection
    #cluster = Cluster(['192.168.1.111'])
    cluster = Cluster(['202.120.40.111'])
    session = cluster.connect('factors') #connect to the keyspace 'factors'

    # get valid stocks in A share,     # IPO PREPARE
    rows = session.execute(''' SELECT stock, ipo_date FROM stock_info WHERE trade_status = '1' ALLOW FILTERING ''')
    stocks = {}
    for row in rows:
        stocks[row.stock] = row.ipo_date

    # sorting factors since they're ordered in cassandra
    # factors = sorted(factors)
    # print("Sorted factors: ", factors)

    #time list
    rows = session.execute('''
        select * from transaction_time 
        where type='M' and time >= %s and time <= %s ALLOW FILTERING;''', [beginDate, endDate])
    dateList = []
    for row in rows:
        dateList.append(row.time)

    # retrieve valid stock number which has been stored in DB at the sorting phase
    countStmt = session.prepare(''' 
    SELECT value from factors_month WHERE stock = 'VALID_STOCK_COUNT' and factor = ? and time = ? ''')
    # prepare SQL
    SQL = "SELECT * FROM "+table+" WHERE stock = ? AND factor IN ("
    for factor in factors:
        SQL = SQL + "'"+ factor + "',"
    SQL = SQL[:-1]
    SQL = SQL +") AND time = ?;"
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), " PREPARE QUERY SQL: \n"+SQL)
    preparedStmt = session.prepare(SQL)
    # select 'trade_status'
    tradeStmt = session.prepare('''select * from factors_month WHERE stock = ?
     and factor = 'trade_status' and time = ? ''')

    # open CSV file & write first line: title
    # NOTICE:  [wb] mode won't result in problem of blank line
    with open(fileName, 'w') as csvFile:
        factors = factors + ['Yield_Rank_Class']
        names = ['id']  + factors # column names
        print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), " ---- Starting to export ------ \r\n")
        f = csv.writer(csvFile, delimiter=',',lineterminator='\n')
        f.writerow(names)

        # retrieve data
        for day in dateList:
            # real factor ranking on every trading day
            factorSizeMap = {}
            for factor in factors:
                rows = session.execute(countStmt,(factor,day))
                for row in rows:
                    factorSizeMap[factor] = row.value
                    break
            for stock, ipo_date in stocks.items():
                ## Trade_Status Filtering
                tradeRow = session.execute(tradeStmt,(stock, day))
                valid = 0
                for status in tradeRow:
                    valid = status.value
                    break
                if valid != 1:
                    continue

                ## Calculating
                line = []
                dic = {}    # paired K/V for ordering
                rows = session.execute(preparedStmt, (stock,day))

                # pass when no data
                empty = True
                line.append(stock+'_' + str(day))
                for row in rows:
                    empty = False
                    # IPO Filtering
                    # filterDate = datetime.datetime.strptime(str(day), "%Y-%m-%d") + relativedelta(months=-3)
                    # ipoDate = datetime.datetime.strptime(str(ipo_date), "%Y-%m-%d")
                    # if (filterDate <= ipoDate):
                    if (day.date() - ipo_date.date()).days <= 92:
                        continue
                    if row.factor.find('rank') != -1:
                        rank = math.ceil(row.value / factorSizeMap[row.factor] * 1000) # normalize rank value

                        if row.factor.find('Yield') != -1:
                            # rank = int(row.value / totalStockNum * 1000)
                            ##################################################
                            ####### CODE Area for Yield Rank Classification ##
                            ##################################################
                            # class 1: [1, 26]
                            if rank > 1 * 10 and rank < 26 * 10:
                                #line.append(1)
                                dic['Yield_Rank_Class'] = '1'
                            # class 0: [74, 99]
                            elif rank > 74 * 10 and rank < 99 * 10:
                                #line.append(0)
                                dic['Yield_Rank_Class'] = '0'
                            else:
                                #line.append('') #no class, fill in empty char to keep CSV well-formed
                                dic['Yield_Rank_Class'] = ''
                            # line.append(rank)

                        dic[row.factor] = rank

                    # elif row.factor.find('Yield') != -1:
                    #     # line.append('') # empty for Yield Binary Class
                    #     # line.append(str(row.value))
                    #     dic['Yield_Rank_Class'] = ''
                    #     dic['Yield'] = str(row.value)
                    else:
                        # line.append(str(row.value))
                        dic[row.factor] = row.value
                if empty:
                    continue
                # write row
                # print (dic)
                empty = False
                for factor in factors:
                    try:
                        line.append(dic[factor])
                    except KeyError:
                        # print(" --- Empty Omitted %s 's factor: %s " % (row.stock, factor))
                        empty = True
                        break
                if empty == False:
                    f.writerow(line)
            print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "  Writing at "+str(day))
    # close connection with cassandra
    cluster.shutdown()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Writing to ',fileName,' complete!')

##############################################
################# USAGE EXAMPLE ##############
# export('D:\\rongshidata\\alldata_416_2.csv', datetime.date(2015,1,31),datetime.date(2017,3,31),factors=['mkt_freeshares_rank', 'mmt_rank', 'roa_growth_rank','Yield_rank'])
export('E:\\train-2017.csv', datetime.date(2017,3,1),datetime.date(2017,4,30),factors=['mkt_freeshares_rank', 'roa_growth_rank', 'mmt_rank', 'mfd_buyamt_d_rank', 'mfd_buyamt_d4_rank','Yield_rank'])

