#!/usr/bin/env python
# coding: utf-8

# In[25]:

###Import Packages########
import random
import math
import numpy as np
from numpy.linalg import inv
from scipy.stats import norm
rv=norm()
import pandas as pd

from numpy import linalg as LA



mat=np.array([[1,0.2,0.04],
            [0.2,1,0.2],
            [0.04,0.2,1]])
w,v=LA.eig(mat)
half=np.dot(v,np.diag(np.sqrt(w)))


def AcceptReject_2(b=6,cov=half): #random sample functions
    while True:
        r=random.uniform(0,1)**(1/3)
        rn=np.random.normal(loc=0.0, scale=1.0, size=3)
        norm_2=np.linalg.norm(rn,ord=2)
        theta=rn/norm_2
        random_unitball=r*theta
        #x = random.uniform(-scale, scale)
        y = random.uniform(0, 1)
        if y*b <= b*(1-np.linalg.norm(random_unitball,ord=2)**2)**(b-1):
            return np.dot(cov,random_unitball)

def AceeptReject(scale=1,c1=15/(8),power=4): #random sample functions
    while True:
        x = random.uniform(-scale, scale)
        y = random.uniform(0, 1)
        if y*c1 <= 15/(8*scale**5)*math.pow((scale-x)*(x+scale), power):
            return 1/2*x


# In[53]:
###########Some needed function classes to solve $\phi^{-1}(x)=0############

class Solution:
    def __init__(self, th=1e-2):
        self.th = th  #threshold  th=1e-4

    
    def phi_root(self,c,d,thetax):
        a_n = c
        b_n = d
        m_n=(a_n+b_n)/2
        iter=0
        while abs(self.phi(m_n,thetax)) >= self.th and iter<=500:  
             #   x = x- stepsize*self.phi(a,x,thetax)/self.phi_p(a,x)
            iter+=1
            m_n = (a_n + b_n)/2
            f_m_n = self.phi(m_n,thetax)
            if self.phi(a_n,thetax)*f_m_n < 0:
                a_n = a_n
                b_n = m_n
            elif self.phi(b_n,thetax)*f_m_n < 0:
                a_n = m_n
                b_n = b_n
            elif f_m_n == 0:
                print("Found exact solution.")
                return m_n
            else:
                print("Bisection method fails.")
                return 0
        return (a_n + b_n)/2
    
    def phi_root_mis(self,c,d,thetax):
        a_n = c
        b_n = d
        m_n=(a_n+b_n)/2
        iter=0
        while abs(self.phi_mis(m_n,thetax)) >= self.th and iter<=50:  #
            iter+=1
            m_n = (a_n + b_n)/2
            f_m_n = self.phi_mis(m_n,thetax)
            if self.phi(a_n,thetax)*f_m_n < 0:
                a_n = a_n
                b_n = m_n
            elif self.phi_mis(b_n,thetax)*f_m_n < 0:
                a_n = m_n
                b_n = b_n
            elif f_m_n == 0:
                #print("Found exact solution.")
                return m_n
            else:
                #print("Bisection method fails.")
                return -2
        return (a_n + b_n)/2
    
    def phi(self,x,thetax):
        
        #return x-(-x**13/13+6*x**11/11-15*x**9/9+20*x**7/7-3*x**5+2*x**3-x+0.34099234)/(1-x**2)**6+thetax
        return x-(-(2*x)**9/9+4*(2*x)**7/7-6*(2*x)**5/5+4*(2*x)**3/3-2*x+128/315)/(2*(1-(2*x)**2)**4)+thetax

    def F(self, x,thetax):
        if x-thetax>0.5:
            return 0
        elif x-thetax<-0.5:
            return x
        else:

        # return x-(-x**13/13+6*x**11/11-15*x**9/9+20*x**7/7-3*x**5+2*x**3-x+0.34099234)/(1-x**2)**6+thetax
            return x*315/256*(-(2*(x-thetax))**9/9+4*(2*(x-thetax))**7/7-6*(2*(x-thetax))**5/5+4*(2*(x-thetax))**3/3-2*(x-thetax)+128/315)
    def phi_mis(self,x,thetax):
        return x-(1-rv.cdf(x))/(rv.pdf(x))+thetax


###################Function classes of kernels and their derivatives#################
class Kernel:
    def __init__(self,v_i,y_t,n,h):
        self.v_i=v_i
        self.y_t=y_t
        self.n=n
        self.h=h#2*n**(-1/5)

    def sec_kernel_h(self,t):
        vec=self.kernel((self.v_i-t)/self.h)
        return 1/(self.n*self.h)*sum(vec*self.y_t)
    
    def sec_kernel_f(self,t):
        vec=self.kernel((self.v_i-t)/self.h)
        return 1/(self.n*self.h)*sum(vec)
    
    def sec_kernel_h1(self,t):
        vec=self.kernel2((self.v_i-t)/self.h)
        return -1/(self.n*self.h**2)*sum(vec*self.y_t)
    
    def sec_kernel_f1(self,t):
        vec=self.kernel2((self.v_i-t)/self.h)
        return -1/(self.n*self.h**2)*sum(vec)
    
    def sec_kernel_h2(self,t):
        vec=self.kernel3((self.v_i-t)/self.h)
        return 1/(self.n*self.h**3)*sum(vec*self.y_t)
    def sec_kernel_f2(self,t):
        vec=self.kernel3((self.v_i-t)/self.h)
        return 1/(self.n*self.h**3)*sum(vec)

    
    def kernel(self, x):
        return (1-x**2)**5*((abs(x)<=1)*1)

    def kernel2(self, x):
        return -5*(1-x**2)**4*2*x*((abs(x)<=1)*1)

    def kernel3(self, x):
        return -5*(1-x**2)**3*(2-18*x**2)*((abs(x)<=1)*1)

    def sec_kernel_whole1(self,t):
        return self.sec_kernel_h(t)/self.sec_kernel_f(t)
    
    def sec_kernel_whole2(self,t):
        return (self.sec_kernel_h1(t)*self.sec_kernel_f(t)-self.sec_kernel_h(t)*self.sec_kernel_f1(t))/self.sec_kernel_f(t)**2
    def sec_kernel_whole3(self,t):
        kh1=self.sec_kernel_h1(t)
        kf0=self.sec_kernel_f(t)
        kh0=self.sec_kernel_h(t)
        kf1=self.sec_kernel_f1(t)
        kh2=self.sec_kernel_h2(t)
        kf2=self.sec_kernel_f2(t)
        
        return ((kh1*kf0)**2-(kh0*kf1)**2-kh0*kh2*kf0**2+kh0**2*kf0*kf2)/(kh1*kf0-kh0*kf1)**2
    def poly_vector(self,t,q):
        vec=self.kernel((self.v_i-t)/self.h)
        p_q=((self.v_i-t)/self.h)**q
        return 1/(self.n*self.h)*sum(vec*self.y_t*p_q)
    
    def poly_deno(self,t,q):
        vec=self.kernel((self.v_i-t)/self.h)
        p_q=((self.v_i-t)/self.h)**q
        return 1/(self.n*self.h)*sum(vec*p_q)
    
    def vec_poly(self,q,t):
        vec=[]
        for i in range(q):
            vec.append(self.poly_vector(t,i))
        return np.array(vec)
    
    def mat_poly(self,q,t):
        mat=np.zeros((q,q))
        for i in range(q):
            for j in range(q):
                mat[i,j]=self.poly_deno(t,i+j)
        return mat
                
    def sol_localpoly(self,q,t):
        vec=np.dot(np.linalg.inv(self.mat_poly(q,t)),self.vec_poly(q,t).reshape(-1,1))
        return vec
    
    def phi(self,t,thetax):
        return t+self.sec_kernel_whole1(t)/self.sec_kernel_whole2(t)+thetax
    
    
    def phi_p(self,t): 
        return 1+self.sec_kernel_whole3(t)
    
    def phi_root(self,y,thetax,stepsize):
        iter=0
        x=(-1.6)*np.exp(y)/(1+np.exp(y))+0.3
        while abs(self.phi(x,thetax)) >= 1e-4 and iter<=1000:  # 判断是否收敛
            y = y- stepsize*self.phi(x,thetax)/(self.phi_p(x)*(-1.6)*np.exp(y)/(1+np.exp(y))**2)
            #print(y)
            x=(-1.6)*np.exp(y)/(1+np.exp(y))+0.3
            iter=iter+1 # (x*x - a)/(2x)
        return x




#Function of the exploration of UCB algorithm
def explore(array_count,array_resp,y_new,p_new,length):
    index=np.floor(p_new/length)
    array_count[index]+=1
    array_resp[index]+=y_new
    return array_count,array_resp#,price,y_t_ucb


######################################
#########Log in real data ############
######################################

Real_data= pd.read_csv("Loan_processed.csv")#real data covariate
#no=pd.read_csv("data_2.csv") #real data noise
#no=np.array(no.T)
#no=no[0]
data_new=Real_data.to_numpy()




data_new=data_new[1:50000,[2,3,4,5]]

mean1=np.mean(data_new,axis=0)
std1=np.std(data_new,axis=0)
data_new2=(data_new-mean1)/std1 #preprocessing, standardize
theta=np.array([-0.3676425,2.428852,0.6608713,-0.64923])



# In[56]:
#############################################################
#########Performance of Bandit algorithm on real dataset######
##############################################################


solu=Solution()
#Basic Setups
T=30000
reg_Tcum_ucb_acc=np.zeros(30000,dtype=int)
reg_Tcum_ucb_acc.reshape(1,30000)

for times in range(10):
    reg_ucb=[]
    t=1
    array_count=np.zeros(10)#Basic setups
    array_resp=np.zeros(10)
    count_bin=10
    length=0.6
    p_t_new=random.uniform(0,6)
    x=data_new2[1,:]
    theta_x=0.3+np.dot(theta,x)
    v=theta_x+AceeptReject(1,15/8,4) #v_t#
    y_t_new=int(int(p_t_new<=v))
    reg_ucb=[]
    reg_ucb_acc=[] #record regret
    while t<=T:
        index=int(np.floor(p_t_new/length))
        array_count[index]+=1.0
        array_resp[index]+=y_t_new
        array_new=np.zeros(10)
        for i in range(10):
            array_new[i]=np.mean(array_resp[i])+np.sqrt(1/(array_count[i]+1))
        p_t_new=np.argmax(array_new)*length #post price p_t
        #x=np.array(AcceptReject_2(4,half))
        x=data_new2[t+1,:] #login new covariate
        theta_x=0.3+np.dot(theta,x)
        v=theta_x+AceeptReject(1,15/8,4)
        y_t_new=int(int(p_t_new<=v))
        true=solu.phi_root(-0.49,0.49,theta_x) #optimal price
        op_price=true+theta_x
        rev_diff=solu.F(op_price,theta_x)-solu.F(p_t_new,theta_x) #revenue difference
        #print(rev_diff)
        reg_ucb.append(abs(rev_diff))
        np.savetxt("reg_ucb.csv", np.array(reg_ucb), delimiter=",")
        reg_ucb_acc.append(sum(reg_ucb))
        np.savetxt("reg_ucb_acc.csv", np.array(reg_ucb_acc), delimiter=",") #regret of bandit algorithm
        t=t+1

    #update
    reg_ucb_acc=np.array(reg_ucb_acc)
    reg_Tcum_ucb_acc=np.vstack([reg_Tcum_ucb_acc,reg_ucb_acc.reshape(1,30000)]) 
    np.savetxt("reg_Tcum_ucb_acc.csv", np.array(reg_Tcum_ucb_acc), delimiter=",")
    

