import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import random

ctq = random.randrange(2,5)

X,y = make_blobs(n_samples = 50,centers=5,n_features=2)

#X = np.array([[1,2],[1.5,1.6],[5,8],[1,0.6],[8,8],[9,11],[8,2],[9,3],[10,3]])
colors = 10*['g','r','c','k','b']

class MeanShift:
	def __init__(self,radius=None,rad_norm_step = 100):
		self.radius = radius
		self.rad_norm_step = rad_norm_step
	def fit(self,data):
		if self.radius == None:
			all_data_ctd = np.average(abs(data),axis = 0)
			all_data_norm = np.linalg.norm(abs(all_data_ctd))
			self.radius = all_data_norm/self.rad_norm_step
		centroids = {}
		for i in range(len(data)):
			centroids[i]=data[i]
		while True:
			n_centroids = []
			for i in centroids:
				wts = [i for i in range(self.rad_norm_step)][::-1]
				in_bdwth = []
				centroid = centroids[i]
				for ftst in data:
					dist =  np.linalg.norm(ftst-centroid)
					if dist == 0:
						dist =0.00000000001
					wt_idx = int(dist/self.radius)
					if wt_idx > self.rad_norm_step-1:
						wt_idx = self.rad_norm_step-1
					to_add = (wts[wt_idx]**2)*[ftst]
					in_bdwth +=to_add
				n_centroid = np.average(in_bdwth,axis=0)
				n_centroids.append(tuple(n_centroid))
			unq = sorted(list(set(n_centroids)))
			to_pop = []
			for i in range(len(unq)):
				for ii in range(1,len(unq)):
					if unq[i] == unq[ii]:
						pass
					elif np.linalg.norm(np.array(unq[i])-np.array(unq[ii])) <=self.radius:
						to_pop.append(unq[ii])
						break
			for i in to_pop:
				try:
					unq.remove(i)
				except:
					pass
			prev_centroids = dict(centroids)
			centroids = {}
			for i in range(len(unq)):
				centroids[i] = np.array(unq[i])
			optimised = True
			for i in centroids:
				if not np.array_equal(centroids[i],prev_centroids[i]):
					optimised = False
				if not optimised:
					break
			if optimised:
				break
		self.centroids = centroids;
		self.classifications = {}
		for i in range(len(self.centroids)):
			self.classifications[i] = []
		for ftst in data:
			dist = [np.linalg.norm(ftst-self.centroids[c]) for c in self.centroids]
			classification = dist.index(min(dist))
			self.classifications[classification].append(ftst)


	def predict(self,data):
		dist = [np.linalg.norm(ftst-self.centroids[c]) for c in self.centroids]
		return dist.index(min(dist))

clf = MeanShift()
clf.fit(X)
ctds = clf.centroids

for cl in clf.classifications:
	clo = colors[cl]
	for ftst in clf.classifications[cl]:
		plt.scatter(ftst[0],ftst[1],color = clo,s=150)

for c in ctds:
	plt.scatter(ctds[c][0],ctds[c][1],marker='*',s=150)
print(ctds)
plt.show()