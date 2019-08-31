from statistics import mean
import numpy as np
import matplotlib.pyplot as pt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6],dtype = np.float64)
ys = np.array([7,4,5,2,0,1],dtype = np.float64)

def create_dset(n,variance,step = 3,corr = True):
	val = 1
	ys=[]
	xs=[]
	for i in range(n):
		y = val + random.randrange(-variance,variance)
		ys.append(y)
		if( corr ):
			val+=step
		else:
			val = val-step
		xs.append(i)
	return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)

xs,ys = create_dset(40,10)

def best_fit_slope(xs,ys):
	m = (mean(xs)*mean(ys))
	m = m - mean(xs*ys);
	m = m / ((mean(xs)*mean(xs)) - mean(xs*xs))
	return m

m = best_fit_slope(xs,ys) 

pt.scatter(xs,ys)

def y_intercept(xs,ys,m):
	b = mean(ys) - (m*mean(xs))
	return b

b = y_intercept(xs,ys,m)

reg_line = [(m*x)+b for x in xs]

def err_sq(ys_o,ys_l):
	return sum((ys_l-ys_o)**2)

def cof_det(ys_o,ys_l):
	ym = [mean(ys_o) for y in ys_o]
	ers = err_sq(ys_o,ys_l)
	erm = err_sq(ys_o,ym)
	return (1-(ers/erm))

r_s = cof_det(ys,reg_line)

print(r_s)

pt.plot(xs,reg_line,color='r')
pt.show()