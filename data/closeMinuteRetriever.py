# pylint: disable=I0011,C0103,C0326,C0301, W0401,W0614
from cassandra.cluster import Cluster
from WindPy import *
import time
import datetime

def minuteRetrieve(startTime, endTime,futures,fields = ['close'], option = "", table = "factors_minute"):
    # Start Wind API
    w.start()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,'---------------- Pulling Started!\n')
    for future in futures:
        # retrieve data
        wsi = w.wsi(future, fields, startTime, endTime, option)
        if wsi.ErrorCode != 0:
            print(future, " !!!===== WIND ERROR CODE: ", wsi.ErrorCode)
            continue

        cluster = Cluster(['192.168.1.111'])
        session = cluster.connect('factors') # factors: factors_month
        sql = "INSERT INTO "+table
        insertPreparedStmt = session.prepare(sql + "(stock, factor, time, value) VALUES (?,?,?,?)")

        # insert into db
        print(wsi)

        for i in range(len(fields)):
            for j in range(len(wsi.Times)):
                session.execute(insertPreparedStmt, (future, fields[i], wsi.Times[j], wsi.Data[i][j]))
                # print(fields[i], wsi.Times[j], wsi.Data[i][j])
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,future,' ---------------- Pulling finished!\n')

minuteRetrieve("2015-04-20 09:00:00", "2015-11-20 15:00:00",
["IF1601.CFE","IF1602.CFE","IF1603.CFE","IF1604.CFE","IF1605.CFE","IF1606.CFE","IF1607.CFE",
"IF1608.CFE","IF1609.CFE","IF1610.CFE","IF1611.CFE","IF1612.CFE","IF1701.CFE","IF1702.CFE","IF1703.CFE"],
['close'])