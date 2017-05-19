# -*- coding: UTF-8 -*-
# pylint: disable=I0011,C0111, C0103,C0326,C0301, C0304, W0401,W0614
from cassandra.cluster import Cluster
from cassandra.util import Date
import time
import datetime
import math

def delError():
    cluster = Cluster(['192.168.1.111'])
    session = cluster.connect('factors') # factors: factors_month
    updatePreparedStmt = session.prepare('''INSERT INTO factors_month(stock, factor, time, value) VALUES (?,'mmt','2017-03-31',?)''')
    deletePreparedStmt = session.prepare('''DELETE FROM factors_month WHERE stock = ? AND factor = ? AND time = '2017-03-28' ''')
    delete31PreparedStmt = session.prepare('''DELETE FROM factors_month WHERE stock = ? AND factor = 'Yield' AND time = '2017-03-31' ''')
    deleteYieldStmt = session.prepare('''DELETE FROM factors_month WHERE stock = ? AND factor = 'Yield' ''')
    deleteInvalidStmt = session.prepare('''DELETE FROM factors_month WHERE stock = ?  AND factor = ? AND time < ? ''')
    deleteInvalidDayStmt = session.prepare('''DELETE FROM factors_day WHERE stock = ?  AND factor = 'close' AND time < ? ''')
    deleteWrongData =  session.prepare('''DELETE FROM factors_month WHERE stock = ? AND factor = ? AND time = '2017-03-31' ''')
# rows = session.execute(
# ''SELECT * FROM factors_month WHERE factor = 'mmt' AND time = '2017-03-28' ALLOW FILTERING''')
    cnt = 1
# for row in rows:
#     session.execute(updatePreparedStmt,(row.stock,row.value))
#     if cnt % 1000 == 0:
#         print(" Updating %d at %s" % (cnt, row.stock))
#     cnt += 1

# rows = session.execute('''SELECT * FROM factors_month WHERE time = '2017-03-28' ALLOW FILTERING''')
# for row in rows:
#     session.execute(deletePreparedStmt,(row.stock,row.factor))
#     if cnt % 1000 == 0:
#         print(" Delete %d at %s" % (cnt, row.stock))
#     cnt += 1
# rows = session.execute('''select stock from stock_info''')
# for row in rows:
#     # session.execute(delete31PreparedStmt,(row.stock,))
#     session.execute(deleteYieldStmt,(row.stock,))
#     if cnt % 1000 == 0:
#         print(" Deleting %d at %s" % (cnt, row.stock))
#     cnt += 1
# rows = session.execute('''select stock, ipo_date from stock_info where trade_status= '1' ALLOW FILTERING ''')
# columns = ['close','mkt_freeshares','mfd_buyamt_d1', 'mfd_sellamt_d1', 'roa', 'pe', 'pb','mfd_buyamt_d2', 'mfd_sellamt_d2','mfd_buyamt_d4', 'mfd_sellamt_d4','mmt','Yield','mkt_freeshares_rank', 'mmt_rank','roa_growth_rank', 'Yield_rank']
# for row in rows:
#     # session.execute(delete31PreparedStmt,(row.stock,))
#     for factor in columns:
#         session.execute(deleteInvalidStmt,(row.stock, factor, row.ipo_date))
#     if cnt % 1000 == 0:
#         print(" Deleting %d at %s" % (cnt, row.stock))
#     cnt += 1

    # cnt = 1
    # rows = session.execute('''select stock, ipo_date from stock_info where trade_status= '1' ALLOW FILTERING ''')
    # for row in rows:
    #     session.execute(deleteInvalidDayStmt,(row.stock, row.ipo_date))
    #     if cnt % 1000 == 0:
    #         print(" Deleting %d at %s" % (cnt, row.stock))
    #     cnt += 1
    rows = session.execute('''select * from stock_info ALLOW FILTERING ''')
    columns = ['close','mkt_freeshares','mfd_buyamt_d1', 'mfd_sellamt_d1', 'roa', 'pe', 'pb',
    'mfd_buyamt_d2', 'mfd_sellamt_d2','mfd_buyamt_d4', 'mfd_sellamt_d4','mmt','Yield', 'roa_growth', 
    'mkt_freeshares_rank', 'mmt_rank','roa_growth_rank', 'Yield_rank']
    print ("deleting : ", columns)
    for row in rows:
        # print(row.stock,  row.ipo_date)
        # session.execute(delete31PreparedStmt,(row.stock,))
        # for factor in columns:
        #     session.execute(deleteWrongData,(row.stock, factor))
        if cnt % 1000 == 0:
            print(" Deleting %d at %s" % (cnt, row.stock))
        cnt += 1
    print(" When Iteration Finished Once")
    for row in rows:
        print(", Can not Repeat!!!")
        if cnt % 1000 == 0:
            print(" Deleting %d at %s" % (cnt, row.stock))
        cnt += 1
    cluster.shutdown()
delError()