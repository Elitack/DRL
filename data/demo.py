# cassandra connection
from cassandra.cluster import Cluster
import matplotlib.pyplot as plt

cluster = Cluster(['202.120.40.111'])
session = cluster.connect('factors') #connect to the keyspace 'factors'
preparedStmt = session.prepare('''INSERT INTO factors_minute(stock, factor, time, value) VALUES (?,?,?,?)''')

stockList = ['IF1601.CFE', 'IF1602.CFE', 'IF1603.CFE', 'IF1604.CFE', 'IF1605.CFE', 'IF1606.CFE', 'IF1607.CFE', 'IF1608.CFE', 'IF1609.CFE', 'IF1610.CFE', 'IF1611.CFE', 'IF1612.CFE','IF1701.CFE', 'IF1702.CFE', 'IF1703.CFE']
factorList = ['open', 'high', 'low', 'close', 'volume', 'amt', 'chg', 'pct_chg', 'oi', 'MA']

data = {}

for stock in stockList:
    data[stock] = {}
    for factor in factorList:
        #f = open(stock+'_'+factor+'.csv', 'a+')
        data[stock][factor] = []
        rows = session.execute("select * from factors_minute where stock='"+stock+"' and factor = '"+factor+"' and time < '20" + stock[2:4] + "-" + stock[4:6] + "-25' ALLOW FILTERING;")
        for row in rows:
            data[stock][factor].append(row.value)
    print stock + ' crawling done'
        #    f.write(str(time.year)+','+str(time.month)+','+str(time.day)+','+str(time.hour))
        #    f.write(','+str(time.minute)+','+str(row.value)+'\n')

for stock in stockList:
    plt.figure()
    x_values = range(len(data[stock]["close"]))
    y_values = data[stock]["close"]
    plt.plot(x_values, y_values)
    plt.savefig(stock+'.png')

#print 'parsing begin'
#for stock in stockList:
 #   f = open(stock+'.csv', "a+")
  #  for factor in factorList:
  #      for index in range(len(data[stock][factor])):
  #          f.write(str(index))
   #         for factorSecond in factorList:
   #             f.write(',' + str(data[stock][factorSecond][index]))
   #         f.write('\n')
 #   print stock + ' parsing done'
  #  f.close()


# close connection with cassandra
#cluster.shutdown()




