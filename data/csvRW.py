import csv
with open('E:\\eggs.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['Spam','Lovely Spam','Wonderful Spam'])

# with open('E:\\names.csv', 'w') as csvfile:
#     fieldnames = ['first_name', 'last_name']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
#     writer.writerow({'last_name': 'Lovely'})
#     writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})

# with open('E:\\names.csv') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         print(row['first_name'], row['last_name'])