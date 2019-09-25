import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1,2],[1.5,1.6],[5,8],[1,0.6],[8,8],[9,11]])

clf = KMeans(n_clusters = 6)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_

colors = 10*['g.','r.','c.','k.','b.']

for i in range(len(X)):
	plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize = 10)

plt.scatter(centroids[:,0],centroids[:,1],marker = 'x',s=150,linewidths =5)
plt.show()