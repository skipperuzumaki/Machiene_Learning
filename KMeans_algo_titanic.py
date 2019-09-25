import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_excel('titanic3.xls')
df.drop(['body','name'],1,inplace = True)
df.fillna(0,inplace = True)

def handle_non_numeric(df):
	clm = df.columns.values
	for cl in clm:
		text_digit_vals = {}
		def toint(val):
			return text_digit_vals[val]
		if df.dtypes[cl] != np.int64 and df.dtypes[cl] != np.float64:
			clm_cnt = df[cl].values.tolist()
			un_ele = set(clm_cnt)
			x=0
			for un in un_ele:
				if un not in text_digit_vals:
					text_digit_vals[un] = x
					x+=1

			df[cl] = list(map(toint,df[cl]))
	return df

df = handle_non_numeric(df)

X = np.array(df.drop(['survived'],1).astype(float))
y = np.array(df['survived'])

X = scale(X)

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

correct = 0
for i in range(len(X)):
	predict  =np.array(X[i].astype(float))
	predict = predict.reshape(-1,len(predict))
	pdn = clf.predict(predict)
	if pdn == y[i]:
		correct+=1

print(correct/len(X))