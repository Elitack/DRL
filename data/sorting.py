'''
##Sorting Examples:
students=[('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
#default ascending order
y = sorted(students, key=lambda student: student[2])
print (y,"\nAscending order by age: ",students)

import collections
Student = collections.namedtuple('Student', 'name grade age')
print ("Type of s is : ", type(Student))
s = Student(name = 'john', grade = 'A', age = 15)
print ("%s is %s years old, got an %s in Math" % (s.name, s.age, s.grade))
students=[Student(name = 'john', grade = 'C', age = 15), Student(name = 'Enna', grade = 'B', age = 12), Student(name = 'dave', grade = 'A', age = 10)]
print ("Descending order by age: ", sorted(students, key=lambda x: x.age, reverse = True))
print ("Ascending order by grade: ", sorted(students, key=lambda x: x.grade))

from operator import itemgetter, attrgetter
print ("Descending order by grade: ", sorted(students, key = itemgetter(1)))
print ("Ascending order by name: ", sorted(students, key = attrgetter('name')))
'''
# pylint: disable=I0011,C0103,C0326,C0301, W0401,W0614
from cassandra.cluster import Cluster
from cassandra.util import Date
import datetime
import time

#sorting factors we need
# 1. get all the transaction date
# 2. for each time point, select all stock's factor value
# 3. sorting
# 4. index them & insert into DB
def sort_factors(beginDate, endDate=datetime.datetime.today().date(), factors = [], table = "factors_month", descending=True):
    if len(factors) == 0:
        return
    #cassandra connection
    cluster = Cluster(['192.168.1.111'])
    session = cluster.connect('factors') #connect to the keyspace 'factors'
    # get transaction date monthly
    rows = session.execute('''
        select * from transaction_time 
        where type='M' and time >= %s and time <= %s ALLOW FILTERING;''', [beginDate, endDate])
    dateList = []
    for row in rows:
        dateList.append(row.time)
    print(str(dateList))
    # prepare, Only Sorting Non-NaN Data
    selectPreparedStmt = session.prepare(
        "select * from " + table + " where factor=? and time=? and value < NaN ALLOW FILTERING")
    insertPreparedStmt = session.prepare(
        "INSERT INTO " + table + "(stock, factor, time, value) VALUES (?,?,?,?)")
    # select 'trade_status'
    tradeStmt = session.prepare('''select * from factors_month WHERE stock = ?
     and factor = 'trade_status' and time = ? ''')

    # IPO map
    ipoMap = {}
    rows = session.execute(''' SELECT stock, ipo_date FROM stock_info WHERE trade_status = '1' ALLOW FILTERING ''') 
    for row in rows:
        ipoMap[row.stock] = row.ipo_date
    # sort each factor for all stocks at each time step
    for factor in factors:
        print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), " Sorting [ %s ] started !" % (factor))
        for day in dateList:
            rows = session.execute(selectPreparedStmt, (factor, day))
            # 无数据跳过
            empty = True
            for x in rows:
                empty = False
                break
            if empty:
                continue
            #print (rows.currentRows)
            sortedRows = sorted(rows, key=lambda x: x.value, reverse=descending)
            cnt = 0
            rank = 0
            prev = -1000000000000
            for row in sortedRows:
                ###############################
                ### Filter out invalid data ###
                ##### VALID: Data's date > IPO + 92 && trade_status = 1 ###
                try:
                    if (day.date() - ipoMap[row.stock].date()).days <= 92:
                        continue
                except KeyError as e:
                    print(" Invalid stock: ", e)
                tradeRow = session.execute(tradeStmt,(row.stock, day))
                valid = 0
                for status in tradeRow:
                    valid = status.value
                    break
                if valid != 1:
                    continue
                cnt += 1
                # same value, same rank
                if row.value != prev:
                    rank += 1
                    prev = row.value
                session.execute_async(insertPreparedStmt, (row.stock, factor + '_rank', row.time, rank))
                # if cnt < 20:
                #     print(row.time,factor, row.stock, ' ', row.value, ' ',  cnt)
                # # if row.stock == '600651.SH':
                # if row.stock == '603636.SH':
                #     print("--- value --",row.time,factor, row.stock, ' ', row.value, ' rank ',rank, cnt)
            ## Store Valid Stock Number for Ranking Use
            session.execute(insertPreparedStmt,('VALID_STOCK_COUNT',factor + '_rank', day, rank))
            print("%s - [ %s ] - complete sorting [ %d stocks], total rank %d" % (day.date().strftime("%Y-%m-%d"), factor, cnt, rank))
        once = False
    # close connection with cassandra
    #cluster.shutdown()

##############################################
################ Invoke Function #############
#sort_factors("2009-01-01", factors=['mkt_freeshares','mmt','roa_growth','mfd_buyamt_d1', 'mfd_sellamt_d1', 'roa', 'pe', 'pb','mfd_buyamt_d2', 'mfd_sellamt_d2','mfd_buyamt_d4', 'mfd_sellamt_d4'])
#sort_factors("2009-01-01", factors=['Yield'])
# sort_factors("2009-01-01", factors=['Yield'])
# sort_factors("2017-03-31", endDate="2017-03-31", factors=['Yield'])
# sort_factors("2015-08-31",endDate="2015-08-31", factors=['mkt_freeshares','mmt','roa_growth','Yield'])
# sort_factors("2016-10-31",endDate="2016-10-31", factors=['mkt_freeshares','mmt','roa_growth','Yield'])
# sort_factors("2015-01-01",endDate="2017-04-30", factors=['roa_growth'])
# sort_factors(datetime.date(2017,3,31), datetime.date(2017,4,30), factors=['mfd_buyamt_d','mfd_buyamt_d4','mkt_freeshares','mmt','Yield'])
# sort_factors(datetime.date(2017,3,31), datetime.date(2017,4,30), factors=['Yield'])
# sort_factors("2016-10-31",endDate="2015-08-31", factors=['mmt'])
# sort_factors(datetime.date(2017,3,31), datetime.date(2017,4,30), factors=['mfd_buyamt_d','mfd_buyamt_d4','mkt_freeshares','mmt','Yield'])
sort_factors(datetime.date(2017,3,31), datetime.date(2017,4,30), factors=['Yield'])
