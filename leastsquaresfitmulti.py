import numpy as np
import matplotlib.pyplot as plt

class least_squares_multi(object):
    def __init__(self, datafile):
        self.x, self.y, self.sigma=np.loadtxt(datafile, unpack=True)
    #n=number of paramters
    def parameters(self,n):
        yvec=np.zeros(shape=len(self.y))
        kvec=np.zeros(shape=(len(self.y),n))

        for i in range(len(self.y)):
            yvec[i]=(self.y[i]/self.sigma[i])
        for i in range(len(self.y)):
            kvec[i][0]=(1/self.sigma[i])
        for i in range(len(self.y)):
            kvec[i][1]=(self.x[i]/self.sigma[i])
        for i in range(len(self.y)):
            kvec[i][2]=(self.x[i]**2/self.sigma[i])
    
        kvect=np.transpose(kvec)
        kvec=np.mat(kvec)
        kvect=np.mat(kvect)
        yvec=np.mat(yvec)
        yvec=np.transpose(yvec)
        avec=((kvect*kvec))**(-1)*kvect*yvec             

        Avec=((kvect*kvec))**(-1)*kvect
        Avect=np.transpose(Avec)
        Avec=np.mat(Avec)
        Avect=np.mat(Avect)
        var=Avec*Avect
        self.var=var
        var=np.mat(var)    
        #return the three parameters and their variance
        self.a=avec[2]
        self.b=avec[1]
        self.c=avec[0]
        self.sigmaa=var[0,0]
        self.sigmab=var[1,0]
        self.sigmab=var[2,0]

        #create model
        model=np.zeros(shape=len(self.x))
        for i in range(len(self.x)):
            model[i]=self.a*self.x[i]**2+self.b*self.x[i]+self.c
    
        #plot the data point, the error bars, and the best fit line
        plt.figure(1)
        plt.plot(self.x,self.y,'o')
        plt.errorbar(self.x,self.y,self.sigma,fmt='o')
        plt.plot(self.x, model)



    #method calculates the expected value, then checks the model by running pertubations on the model
    def expected(self,x):

        A=(1,x,x**2)
        A=np.mat(A)
        At=np.transpose(A)
        At=np.mat(At)
        #calculated expected x based off model 
        self.expx=self.a*8**2+self.b*8+self.c   
        #uncertainty in x expectation
        self.sigmax=(A*self.var*At)

        #calculating the correlation coefficient matrix
            
        coeff=self.var
        coeff[0,0]=coeff[0,0]/self.var[0,0]
        coeff[1,1]=coeff[1,1]/self.var[1,1]  
        coeff[2,2]=coeff[2,2]/self.var[2,2]  
        coeff[0,1]=coeff[0,1]/(self.var[0,0]*self.var[1,1])
        coeff[0,2]=coeff[0,2]/(self.var[0,0]*self.var[2,2])  
        coeff[1,0]=coeff[1,0]/(self.var[0,0]*self.var[1,1])
        coeff[1,2]=coeff[1,0]/(self.var[1,1]*self.var[2,2]) 
        coeff[2,0]=coeff[2,0]/(self.var[0,0]*self.var[2,2])  
        coeff[2,1]=coeff[2,0]/(self.var[1,1]*self.var[2,2])  


#peforming singular value decomposition on the coefficient matrix
        U, s, V= np.linalg.svd(coeff, full_matrices=True)

#insert D in as a diagonal to a zeros matrix
        D=np.zeros((3,3))
        D[0,0]=s[0]
        D[1,1]=s[1]
        D[2,2]=s[2]



#generating NX3 random number matrix
        R=np.random.randn(1000,3)
        R=np.mat(R)
        D=np.mat(D)
        V=np.mat(V)
        P=R*np.sqrt(D)*V

        #perturbing the model parameters, and producing a histogram which shows distribution of expected value
        aprime=np.zeros(1000)
        bprime=np.zeros(1000)
        cprime=np.zeros(1000)
        yprime=np.zeros(1000)
        for i in range(len(aprime)):
            aprime[i]=self.a+P[i,0]
            bprime[i]=self.b+P[i,1]
            cprime[i]=self.c+P[i,2]
            yprime[i]=aprime[i]*8**2+bprime[i]*8+cprime[i]
        plt.figure(2)    
        plt.hist(yprime,100)

        #trial curves for  0<x<10
        plt.figure(3)

        xtrial=np.linspace(0,10,10)
        plt.figure(3)
        modeltrial=np.zeros(10)
        for i in range(0,100):
            for j in range(len(xtrial)):
                modeltrial[j]=aprime[i]*xtrial[j]**2+bprime[i]*xtrial[j]+cprime[i]
            plt.plot(xtrial, modeltrial)




