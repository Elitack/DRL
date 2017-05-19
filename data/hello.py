import datetime
begin = datetime.date(2014, 6, 1)
end = datetime.datetime.now()
end = datetime.date(end.year, end.month, end.day)
d = begin
delta = datetime.timedelta(days=1)
while d <= end:
    print(d.strftime("%Y-%m-%d"))
    d += delta
print(end)
