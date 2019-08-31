import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-9999,inplace = True)
df.drop(['id'] , 1 , inplace = True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

acc = clf.score(X_test,y_test)

print(acc)

eg = np.array([[4,2,1,1,1,2,3,2,1],[4,2,3,1,5,2,3,2,1]])
eg = eg.reshape(len(eg),-1)

pre = clf.predict(eg)
print(pre)