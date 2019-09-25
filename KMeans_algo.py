import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1,2],[1.5,1.6],[5,8],[1,0.6],[8,8],[9,11]])
'''
clf = KMeans(n_clusters = 6)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_
'''
colors = 10*['g','r','c','k','b']

class KMean:
	def __init__(self,k = 2,tol = 0.001,max_iter = 300):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter
	def fit(self,data):
		self.centroids = {}
		for i in range(self.k):
			self.centroids[i] = data[i]
		for i in range(self.max_iter):
			self.classifications = {}
			for i in range(self.k):
				self.classifications[i] = []
			for featureset in data:
				distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)
			prev_centroids = dict(self.centroids)
			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification],axis=0)
				optimized = True
			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]
				if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
					print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
					optimized = False
			if optimized:
				break

	def predict(self,data):
		distances = [np.linalg.norm(data-c) for c in self.centroids]
		cffn = distances.index(min(distances))
		return cffn

clf = KMean()
clf.fit(X)

for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker='o',color = 'k',s=150,linewidths=5)

for cffn in clf.classifications:
	color =colors[cffn]
	for ftst in clf.classifications[cffn]:
		plt.scatter(ftst[0],ftst[1],marker='x',color = colors[cffn%5],s=150,linewidths=5)

plt.show()