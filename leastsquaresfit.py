import numpy as np
import matplotlib.pyplot as plt

class least_squares(object):
		def __init__(self,datafile):
			self.x, self.y, self.sigma= np.loadtxt(datafile, unpack=True)

		def expectation(self,nsteps, min, max):
			a=np.linspace(min,max,num=nsteps)
			b=np.linspace(min,max,num=nsteps)
			da=abs(max-min)/float(nsteps)
			db=abs(max-min)/float(nsteps)
			self.expa=0
			self.expb=0
			self.prob = np.zeros(shape=(nsteps,nsteps))
			for i in range(len(a)):
				for j in range(len(b)):
					#reset the probability function value with each loop
					probfxn = 1
					#for each (a,b), calcualte the probability by looping through the data x, y, and sigma
					for k in range(len(self.x)):
						probfxn *= (1/(np.sqrt(2*3.14)*self.sigma[k]))*np.exp(-((self.y[k]-(a[i]*self.x[k]+b[j]))**2)/(2*self.sigma[k]**2))
					#create a matrix with all the probabilty values stored, the first dimension is the a value the second is the b
					self.prob[i][j]=probfxn
			#calculate the expectation of a and b by integrating the probablity matrix
			vol=0
			for i in range(len(a)):
				for j in range(len(b)):
					vol+=db*self.prob[i][j]*da
			self.vol=vol
			for i in range(len(a)):
				for j in range(len(b)):
					self.expa += a[i]*self.prob[i][j]*db*da/self.vol 
					self.expb += b[i]*self.prob[j][i]*db*da/self.vol

        	#calculate the vairance of a and the variance of b
			siga=0
			sigb=0
			for i in range(len(a)):
				for j in range(len(b)):
					siga += (a[i]-self.expa)**2*self.prob[i][j]*db*da/self.vol
					sigb += (b[i]-self.expb)**2*self.prob[j][i]*db*da/self.vol 
			self.siga=np.sqrt(siga)
			self.sigb=np.sqrt(sigb)
        	#calculate the covariance of a and b
			covab=0
			for i in range(len(a)):    
				for j in range(len(b)):
					covab += (a[i]-self.expa)*(b[j]-self.expb)*self.prob[i][j]*db*da/self.vol 
			self.covab=covab

        	#calculate the correlation coefficient
			self.pco=self.covab/(self.siga*self.sigb)
		def plot_model(self):
        	#plot data
			plt.plot(self.x,self.y,'o')
        	#plot model
			xvar=np.linspace(0,10,100)
			model=(self.expa)*xvar+(self.expb)
			return plt.plot(xvar, model)

