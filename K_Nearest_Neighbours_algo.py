import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-9999,inplace = True)
df.drop(['id'] , 1 , inplace = True)
fd = df.astype(float).values.tolist()
random.shuffle(fd)

testsize = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
train_data = fd[:-int(testsize*len(fd))]
test_data = fd[-int(testsize*len(fd)): ]

for i in train_data:
	train_set[i[-1]].append(i[:-1])

for i in test_data:
	test_set[i[-1]].append(i[:-1])

def K_NN(data,predict,k=3):
	if(len(data)>=k):
		warnings.warn('K less than total voting groups')

	dist = []
	for grp in data:
		for ft in data[grp]:
			disq = np.linalg.norm(np.array(ft,dtype = np.float64)-np.array(predict,dtype = np.float64))
			dist.append([disq,grp])
	votes = [i[1]for i in sorted(dist)[:k]]
	vr = Counter(votes).most_common(1)[0][0]
	conf = Counter(votes).most_common(1)[0][1]/k
	return vr , conf

correct = 0
total = 0

for grp in test_set:
	for data in test_set[grp]:
		vote , conf = K_NN(train_set,data,k=7)
		if grp == vote:
			correct+=1
		else:
			print(conf)
		total +=1
print(correct/total)
