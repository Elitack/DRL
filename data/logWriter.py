# pylint: disable=I0011,C0103,C0326,C0301, W0401,W0614
import time
import os
from cassandra.cluster import Cluster

def log(filename, content):
    # open file if exist else create it, close properly even exception occurs
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != os.errno.EEXIST:
                raise
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        file.write(content)

def persist(logDir, session):
    fileList = os.listdir(logDir)#列出目录下的所有文件和目录
    # cluster = Cluster(['192.168.1.111'])
    # session = cluster.connect('factors') # factors: factors_month
    for file in fileList:
        filename = logDir+"\\"+file
        if os.path.isdir(file) is False:
            try:
                with open(filename, 'r') as f:
                    allLines = f.readlines()
                    for line in allLines:
                        session.execute_async(line)
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "Persist complete in ", filename)
            except PermissionError as e:
                print(" --- Error in ",filename,e)

# log("E:\\test\\b\\a.txt", "Insert into a")
# persist("G:\\log\\daily_mfd_buyamt_d")