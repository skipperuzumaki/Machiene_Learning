import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_excel('titanic3.xls')
odf = pd.DataFrame.copy(df)
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

clf = MeanShift()
clf.fit(X)

label = clf.labels_
clssen = clf.cluster_centers_

odf['clg'] = np.nan

for i in range(len(X)):
	odf['clg'].iloc[i] = label[i]

n_cls = len(np.unique(label))
sr = {}
for i in range(n_cls):
	temp = odf[(odf['clg'] == float(i))]
	scls = temp[(temp['survived']==1)]
	s = len(scls)/len(temp)
	sr[i] = s

print(sr)