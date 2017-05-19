
# cassandra connection
from cassandra.cluster import Cluster
import os
import datetime

if not os.path.exists('compositeFactor'):
    os.mkdir('compositeFactor')

# 上市三个月以上
date = datetime.date.today()
month = (date.month + 12 - 3) % 12
if month == 0:
    month = 12
date = datetime.date(date.year, month, date.day).strftime('%Y-%m-%d')

# 接口IP，需要做一个映射
cluster = Cluster(['202.120.40.111'])
session = cluster.connect('factors')  # connect to the keyspace 'factors'

rows = session.execute(
    'select stock from stock_info where ipo_date<\'' + date + '\' ALLOW FILTERING;')

preparedStmt = session.prepare(
    "select * from factors_month where stock=? and factor=? and time>'2012-01-01'")

# 计算mfd_transamt_d1
count = 0
for i in range(len(rows.current_rows)):
    code = rows.current_rows[i].stock
    mfd_buyamt_d1 = session.execute(preparedStmt, (code, 'mfd_buyamt_d1'))
    mfd_sellamt_d1 = session.execute(preparedStmt, (code, 'mfd_sellamt_d1'))
    with open('compositeFactor/' + code + '_mfd_transamt_d1.txt', mode='w') as f:
        for i in range(len(mfd_buyamt_d1.current_rows)):
            mfd_transamt_d1 = mfd_buyamt_d1.current_rows[i].value + \
                mfd_sellamt_d1.current_rows[i].value
            f.write("insert into factors_month(stock,factor,time,value) values ('" + code +
                    "','mfd_transamt_d1','" + str(mfd_buyamt_d1.current_rows[i].time) + "'," + str(mfd_transamt_d1) + ");\r\n")
    count += 1
    print('mfd_transamt_d1 ' + code + ' complete, completed: ' + str(count))

# mfd_transamt_d4
count = 0
for i in range(len(rows.current_rows)):
    code = rows.current_rows[i].stock
    mfd_buyamt_d4 = session.execute(preparedStmt, (code, 'mfd_buyamt_d4'))
    mfd_sellamt_d4 = session.execute(preparedStmt, (code, 'mfd_sellamt_d4'))
    with open('compositeFactor/' + code + '_mfd_transamt_d4.txt', mode='w') as f:
        for i in range(len(mfd_buyamt_d4.current_rows)):
            mfd_transamt_d4 = mfd_buyamt_d4.current_rows[i].value + \
                mfd_sellamt_d4.current_rows[i].value
            f.write("insert into factors_month(stock,factor,time,value) values ('" + code +
                    "','mfd_transamt_d4','" + str(mfd_buyamt_d4.current_rows[i].time) + "'," + str(mfd_transamt_d4) + ");\r\n")
    count += 1
    print('mfd_transamt_d4 ' + code + ' complete, completed: ' + str(count))

# close connection with cassandra
cluster.shutdown()
