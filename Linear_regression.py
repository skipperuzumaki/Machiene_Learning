import pandas as ps
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get("WIKI/GOOGL")

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Low'] *100.0

df['PCT'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Close'] *100.0

df = df[['Adj. Close','HL_PCT','PCT','Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
print("predicted forecast : ",forecast_out)
forecast_out = int(input("enter forecast date : "))
train = int(input("enter 1 to freshly train classifier : "))

print(forecast_out)

df['Label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['Label'],1))

X = preprocessing.scale(X)

X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace = True)
y = np.array(df['Label'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
if train == 1:
	try:
		pickle_in = open('LR.pickle','rb')
		clf = pickle.load(pickle_in)
	except:
		clf = LinearRegression(n_jobs = -1)
		clf.fit(X_train,y_train)
		with open("LR.pickle",'wb') as f:
			pickle.dump(clf,f)
else:
	clf = LinearRegression(n_jobs = -1)
	clf.fit(X_train,y_train)
	with open("LR.pickle",'wb') as f:
		pickle.dump(clf,f)

acc = clf.score(X_test,y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set,acc,forecast_out)

df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix+=one_day
	df.loc[next_date] = [np.nan for _ in range (len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
