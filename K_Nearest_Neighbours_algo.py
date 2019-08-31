import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
style.use('ggplot')

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_fet = [5,7]

[[plt.scatter(ii[0],ii[1],s=100,color = i) for ii in dataset[i]] for i in dataset]

plt.scatter(new_fet[0],new_fet[1],s=100,color = 'b')

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
	return vr

res = K_NN(dataset,new_fet)
print(res)
plt.show()