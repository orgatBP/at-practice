
# -*- coding: utf-8 -*-

MAX = 1000
print(2, end=' ')
aList = []
for i in range(0, MAX+1):
    aList.append(0)

for i in range(3, MAX+1, 2):
    for j in range(i+i, MAX+1, i):
        aList[j] = 1

for i in range(3, MAX+1, 2):
    if(aList[i]==0):
        print(i, end=' ')

