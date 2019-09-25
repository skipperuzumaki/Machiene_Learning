import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from mlxtend.data import loadlocal_mnist

X,y = loadlocal_mnist(images_path='train-images-idx3-ubyte',labels_path='train-labels-idx1-ubyte')

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2)

print(X_train)
print(y_train)

clf = svm.SVC(kernel = 'poly',degree = 5, gamma = 'auto')
clf.fit(X_train,y_train)

acc = clf.score(X_test,y_test)

print(acc)