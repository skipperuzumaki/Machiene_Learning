import numpy as np
import matplotlib.pyplot as pyt
from matplotlib import style
style.use('ggplot')

class SVM:
	def __init__(self,visulasiation = True):
		self.visual = visulasiation
		self.color = {1:'b',-1:'r',0:'y'}
		if self.visual:
			self.fig = pyt.figure()
			self.ax = self.fig.add_subplot(1,1,1)

	def fit(self,data):
		self.data = data
		opt_dict = {}
		transforms = [[1,1],[1,-1],[-1,1],[-1,-1]]
		all_data = []
		for yi in self.data:
			for ft in self.data[yi]:
				for f in ft:
					all_data.append(f)
		self.max_ft = max(all_data)
		self.min_ft = min(all_data)
		all_data = None
		step_size = [self.max_ft*0.1,self.max_ft*0.01,self.max_ft*0.001,self.max_ft*0.0001]
		b_rng_multple = 50
		b_mtpl = 1
		ltst_optm = self.max_ft*10
		for step in step_size:
			w = np.array([ltst_optm,ltst_optm])
			optimized = False
			while not optimized:
				for b in np.arange(-1*self.max_ft*b_rng_multple,self.max_ft*b_rng_multple,step*b_mtpl):
					for transform in transforms:
						w_t = w*transform
						fnd_opt = True
						for i in self.data:
							for xi in self.data[i]:
								yi = i
								if not yi*(np.dot(w_t,xi)+b) >=1:
									fnd_opt =False
						if fnd_opt:
							opt_dict[np.linalg.norm(w_t)] = [w_t,b]
				if w[0] < 0 and w[1] < 0:
					optimized = True
					print('optimized step')
				else:
					w = w-step
			norms = sorted([n for n in opt_dict])
			opt_choice = opt_dict[norms[0]]
			self.w = opt_choice[0]
			self.b = opt_choice[1]
			ltst_optm = opt_choice[0][0]+step*2

			for i in self.data:
				for xi in self.data[i]:
					print(xi,':',i*(np.dot(self.w,xi)+self.b))

	def predict(self,feat):
		cl = (feat[0]*float(self.w[0])+feat[1]*float(self.w[1]))+self.b
		if cl>0:
			classfn = 1
		elif cl<0:
			classfn = -1
		else:
			classfn = 0
		if self.visual:
			self.ax.scatter(feat[0],feat[1],marker = '*',s=200,c=self.color[classfn])
		return classfn

	def visualise(self):
		[[self.ax.scatter(x[0],x[1],s=100,color = self.color[i]) for x in self.data[i]] for i in self.data]
		def hyperplane(x,w,b,v):
			return (-w[0]*x-b+v)/w[1]
		datarange  = [self.min_ft*0.9,self.max_ft*1.1]
		psv1 =hyperplane(datarange[0],self.w,self.b,1)
		psv2 =hyperplane(datarange[1],self.w,self.b,1)
		self.ax.plot(datarange,[psv1,psv2],'k')
		nsv1 =hyperplane(datarange[0],self.w,self.b,-1)
		nsv2 =hyperplane(datarange[1],self.w,self.b,-1)
		self.ax.plot(datarange,[nsv1,nsv2],'k')
		db1 =hyperplane(datarange[0],self.w,self.b,0)
		db2 =hyperplane(datarange[1],self.w,self.b,0)
		self.ax.plot(datarange,[db1,db2],'y--')
		pyt.show()


adata = {1:np.array([[1,7],[2,8],[3,8]]),-1:np.array([[5,1],[6,-1],[7,3]])}
predict = [[10,1],[5,-2],[3,4],[2,6],[8,1],[9,0]]
svm =SVM()

svm.fit(data = adata)
for i in predict:
	svm.predict(i)
svm.visualise()
