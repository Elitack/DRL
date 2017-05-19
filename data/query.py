# pylint: disable=I0011,C0103,C0326,C0301, W0401,W0614
from cassandra.cluster import Cluster
from cassandra.util import Date
import time
import datetime
import math

cluster = Cluster(['192.168.1.111'])
session = cluster.connect('factors') #connect to the keyspace 'factors'
rows = session.execute('''SELECT stock, ipo_date FROM stock_info WHERE trade_status = '1' ALLOW FILTERING ''')
stocks = {}
for row in rows:
    stocks[row[0]] = row[1]

sql = "select * from factors_month where stock = ? and factor = 'roa' and time >= '2016-10-31' and time <= '2017-03-31' ALLOW FILTERING"
selectPreparedStmt = session.prepare(sql)
count = 0
res = []
for stock, ipo_date in stocks.items():
    if datetime.date(2016,10,31) < ipo_date.date():
        continue
    rows = session.execute(selectPreparedStmt, (stock, ))
    val = 0
    cnt = 0
    prev = 0
    for row in rows:
        val = row.value
        if cnt == 0:
            prev = val
    if val == float('nan') or prev == val:
        res.append(stock)
    # 1258
print("Length of result: ", len(res), res)
