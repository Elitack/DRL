from cassandra.cluster import Cluster
import datetime

# 读取月数据
TABLE = 'factors_month'
# 从文件中读取因子
f = open('factors.txt', 'r')
# 连接cassandra
cluster = Cluster(['192.168.1.111'])
session = cluster.connect('factors')
# 准备语句
SelectPreparedStmt = session.prepare(
    "select * from " + TABLE + " where factor=? and time=? ALLOW FILTERING")
InsertPreparedStmt = session.prepare(
    "INSERT INTO " + TABLE + "(stock, factor, time, value) VALUES (?,?,?,?)")
while True:
    factor = f.readline()
    # 去掉\r\n，否则查询无结果
    factor = factor.replace('\r', '').replace('\n', '')
    if factor == '':
        break
    # 查询时间
    begin = datetime.date(2009, 1, 1)
    current = datetime.datetime.now()
    current = datetime.date(current.year, current.month, current.day)
    delta = datetime.timedelta(days=1)
    while begin <= current:
        time = begin.strftime("%Y-%m-%d")
        rows = session.execute(SelectPreparedStmt, (factor, time))
        begin += delta
        # 无数据跳过
        empty = True
        for x in rows:
            empty = False
            break
        if empty:
            continue
        # 有数据排序并插入
        rows = sorted(rows, key=lambda x: x.value, reverse=True)
        count = 0
        for row in rows:
            count += 1
            session.execute(InsertPreparedStmt, (row.stock, factor +
                                                 '_rank', row.time, count))
f.close()
