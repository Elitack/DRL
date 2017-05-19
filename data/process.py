# -*- coding: UTF-8 -*-
# pylint: disable=I0011,C0111, C0103,C0326,C0301, C0304, W0401,W0614
from cassandra.cluster import Cluster
from cassandra.util import Date
import time
import datetime
import math

# from correct import delError
from dataMonthRetriever import *
from calculating import *
from sorting import *
from stocks import *
from closeDailyRetriever import *

## Execute the whole process automatically
## 1. retrieve newly updated data from last checkpoint
#########################################################
## Updating Available A share Stock & transaction_time ##
#########################################################
# update transaction time
updateTransactionTime('2017-04-01')
updateTransactionTime('2017-04-01', TYPE='M')
# update stocks
indexes=["000001.SH","399001.SZ",'399006.SZ','000300.SH','000016.SH','000905.SH']
updateAStocks(datetime.date(2017,4,28), extraIndex=indexes)


#########################################################
## Updating Daily Data [close, trade_status, mfdxxx] ####
#########################################################
retrieveSingleFactor('close','2015-01-01',extraIndex=indexes)
dailyRetrieve(datetime.date(2017,4,27), datetime.datetime.today().date(),"G:\\log\\daily_mfd_buyamt_d\\4-28", 
fields1 = ['trade_status','mfd_buyamt_d'],multi_mfd = True)

monthRetrieve(datetime.date(2017,4,1), datetime.date(2017,4,30), 
fields1=['trade_status','close', 'mkt_freeshares','mkt_cap_float','roa'], multi_mfd = False)
# monthRetrieve(datetime.date(2009,1,1), datetime.datetime.today().date(), 
#     fields1=['trade_status','close', 'mkt_freeshares','mkt_cap_float','roa'], multi_mfd = False)


#########################################################
## calculating secondary factors  ####
#########################################################
calculate_mmt(datetime.date(2017,3,31), datetime.date(2017,4,30))
calculate_Yield(datetime.date(2017,3,31), datetime.date(2017,4,30))
calculate_mfd_sum(datetime.date(2017,4,1), datetime.date(2017,4,30),factors=['mfd_buyamt_d','mfd_buyamt_d4'])


# sort_factors("2009-01-01", factors=['mkt_freeshares','mmt','roa_growth','Yield'])
sort_factors(datetime.date(2012,1,1), datetime.date(2017,4,1), factors=['mfd_buyamt_d','mfd_buyamt_d4','mkt_freeshares','mmt','Yield'])