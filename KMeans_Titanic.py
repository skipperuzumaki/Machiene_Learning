import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
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

clf = KMeans(n_clusters = 2)
clf.fit(X)

correct = 0
for i in range(len(X)):
	predict  =np.array(X[i].astype(float))
	predict = predict.reshape(-1,len(predict))
	pdn = clf.predict(predict)
	if pdn[0] == y[i]:
		correct+=1

print(correct/len(X))