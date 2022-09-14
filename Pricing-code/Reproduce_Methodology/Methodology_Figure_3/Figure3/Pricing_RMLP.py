#!/usr/bin/env python
# coding: utf-8


###Import Packages########
import random
import math
import numpy as np
from numpy.linalg import inv
from scipy.stats import norm
rv=norm()
import pandas as pd
from random import sample
from numpy import linalg as LA


mat=np.array([[1,0.2,0.04], #covariate matrix of x#
            [0.2,1,0.2],
            [0.04,0.2,1]])
w,v=LA.eig(mat)
half=np.dot(v,np.diag(np.sqrt(w)))


####random sample functions for covariate and the noise#######
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


# In[78]:
##############################################
############Some needed classes solving \phi^{-1}(x)=0############
##############################################
class Solution:
    def __init__(self, th=1e-2):
        self.th = th  # threshold  th=1e-4
    
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
                #print("Found exact solution.")
                return m_n
            else:
                #print("Bisection method fails.")
                return 0
        return (a_n + b_n)/2
    
    def phi_root_mis(self,c,d,thetax):
        #if self.phi(a,c,thetax)*self(a,d,thetax)>=0:
        #    return None
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
        
        return x-(-(2*x)**9/9+4*(2*x)**7/7-6*(2*x)**5/5+4*(2*x)**3/3-2*x+128/315)/(2*(1-(2*x)**2)**4)+thetax

    def F(self, x,thetax):
        if x-thetax>0.5:
            return 0
        elif x-thetax<-0.5:
            return x
        else:
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
        while abs(self.phi(x,thetax)) >= 1e-4 and iter<=1000:  
            y = y- stepsize*self.phi(x,thetax)/(self.phi_p(x)*(-1.6)*np.exp(y)/(1+np.exp(y))**2)
            #print(y)
            x=(-1.6)*np.exp(y)/(1+np.exp(y))+0.3
            iter=iter+1 # (x*x - a)/(2x)
        return x


# In[80]:


def explore(array_count,array_resp,y_new,p_new,length):
    index=np.floor(p_new/length)
    array_count[index]+=1
    array_resp[index]+=y_new
    return array_count,array_resp#,price,y_t_ucb


# In[81]:
#######################################
############Load real data#############
#######################################

Real_data= pd.read_csv("Loan_processed.csv") #real_data_covariate

#Initialize

data_new=Real_data.to_numpy()
a=list(range(200000))
theta=np.array([-0.3676425,2.428852,0.6608713,-0.64923])
solu=Solution()
B=4


l_0=200
reg_Tcum_acc=np.zeros(25400,dtype=int)
reg_Tcum_tt=np.zeros(25400,dtype=int)
reg_Tcum_tt_acc=np.zeros(25400,dtype=int)
reg_Tcum_tf=np.zeros(25400,dtype=int)
reg_Tcum_tf_acc=np.zeros(25400,dtype=int)
reg_Tcum_bt=np.zeros(25400,dtype=int)
reg_Tcum_bt_acc=np.zeros(25400,dtype=int)
reg_Tcum_mis_acc=np.zeros(25400,dtype=int)


reg_Tcum_acc.reshape(1,25400)
reg_Tcum_tt_acc.reshape(1,25400)
reg_Tcum_tf_acc.reshape(1,25400)
reg_Tcum_bt_acc.reshape(1,25400)
reg_Tcum_mis_acc.reshape(1,25400)

#########Basic Setups#########
T=25400
epi_num=math.ceil(math.log(T/l_0+1, 2))
reg_T=[] #record 30 times
reg_T_tt=[]
reg_T_tf=[]
reg_T_bt=[]
for times in range(10):
    data_new_2=data_new[sample(a,30000),:]                               #input the real data
    data_new_2=data_new_2[:,[2,3,4,5]]
    mean1=np.mean(data_new_2,axis=0)
    std1=np.std(data_new_2,axis=0)
    data_new2=(data_new_2-mean1)#/std1
    
    reg=[]                                                              #for every time, record all the regret
    reg_mis=[]
    reg_tt=[]
    reg_tf=[]
    reg_bt=[]
    reg_acc=[]
    reg_mis=[]
    reg_tt_acc=[]
    reg_tf_acc=[]
    reg_bt_acc=[]
    reg_mis_acc=[]
    y_t_explore=[]
    v_i_explore=[]
    v_i_explore_true=[]
    l_k1_total=0
    for k in range(epi_num): #number of episode
        t=0
        l_k=l_0*2**k
        #if k==0:  ##uncomment when we have uknown m.
        #   m=2
        #else:
        #   m=choose_m(len(y_t_explore),np.array(v_i_explore_true),np.array(y_t_explore))
        #print(m)
        l_k1=math.floor((l_k*2)**(5/7)) ##when m is unknown l_k1=math.floor((l_k*2)**((2*m+1)/(4*m-1))) #m is chosen via choose_m() given in the end. Here we use Pessimistic way to choose m=2. List in our remark 12.
        l_k1_total+=l_k1
        v_i=[]
        v_i_true=[]
        v_t=[]
        y_t=[]
        #y_t_explore=[]
        p_t_coll=[]
        X=np.zeros(3,dtype=int)
        X=X.reshape(1,3)
        para_lk1=math.floor((l_k*2)**(1/2))
        while t<=l_k:                                                               #exploration phase
            if t<=l_k1:  
                x=data_new2[t+l_0*(2**k-1),:]                                       #login data
                theta_x=0.3+np.dot(theta,x)                                          #theta
                v=theta_x+AceeptReject(1,15/8,4)                                    # v_t
                p_t=random.uniform(0,B)                                             #offered p_t
                v_i_true.append(p_t-0.3-np.dot(theta,x))
                p_t_coll.append(p_t)
                op_price=solu.phi_root(-0.49,0.49,theta_x)+theta_x                  #optimal price
                rev_diff=solu.F(op_price,theta_x)-solu.F(p_t,theta_x)               #revenue difference
                print(rev_diff)
                reg.append(abs(rev_diff))
                reg_acc.append(np.nansum(reg))
                np.savetxt("reg.csv", reg, delimiter=",")                            #regret of our algorithm in exploration phase
                np.savetxt("reg_acc.csv", reg_acc, delimiter=",")
                
                reg_mis.append(abs(rev_diff))
                reg_mis_acc.append(np.nansum(reg_mis))
                np.savetxt("reg_mis.csv", reg_mis, delimiter=",")                   #regret of RMLP-2 in exploration phase
                np.savetxt("reg_mis_acc.csv", reg_mis_acc, delimiter=",")

                y_t.append(int(p_t<=v))
                y_t_explore.append(int(p_t<=v))
                v_i_explore_true.append(p_t-0.3-np.dot(theta,x))
                v_t.append(v)
                if t==l_k1: #Update parameters
                    #v_i=np.array(v_i)
                    v_t=np.array(v_t) #vector of vt
                    y_t=np.array(y_t)#vector of y_t
                    p_t_coll=np.array(p_t_coll)

                    v_i_true=np.array(v_i_true)
                    ker=Kernel(v_i_explore_true,y_t_explore,l_k1_total,2*l_k1_total**(-1/5)) 
                    ker_tt=Kernel(v_i_explore_true,y_t_explore,l_k1_total,2*l_k1_total**(-1/5))
                    print('--explore--')
                t=t+1

            else:                                                                                           #exploitation phase
                x=data_new2[t+l_0*(2**k-1),:]
                theta_x=0.3+np.dot(theta,x)
                #print(theta_x)
                true=solu.phi_root(-0.49,0.49,theta_x)
                op_price=true+theta_x #optimal price
                op_rev=solu.F(op_price,theta_x) #optimal revenue
                est_price=ker.phi_root(np.log(-(true-0.3)/(true+1.3)),theta_x,0.5)+theta_x                   #estimated p_t
                rev_diff=op_rev-solu.F(est_price,theta_x) #revenue difference
                print(rev_diff)

                est_price_mis=solu.phi_root_mis(-0.49,0.49,theta_x)+theta_x                                 #estimated price of RMLP-2
                rev_diff_mis=op_rev-solu.F(est_price_mis,theta_x)


                reg.append(abs(rev_diff))                                                                   # our exploitation regret
                reg_acc.append(np.nansum(reg))
                np.savetxt("reg.csv", reg, delimiter=",")                                                    #record our regret
                np.savetxt("reg_acc.csv", reg_acc, delimiter=",")

                reg_mis.append(rev_diff_mis)
                reg_mis_acc.append(np.nansum(reg_mis))
                np.savetxt("reg_mis.csv", reg_mis, delimiter=",")                                            #regret of RMLP-2
                np.savetxt("reg_mis_acc.csv", reg_mis_acc, delimiter=",")

                if t==l_k:
                    print('--exploit--')
                t=t+1
                
    reg=reg[0:T]
    reg_acc=reg_acc[0:T]

    if len(reg_mis)<25400:
        extra=[0]*(25400-len(reg_mis))
        reg_mis=extra.extend(reg_mis)
    reg_mis=reg_mis[0:T]
    if len(reg_mis_acc)<25400:
        extra=[0]*(25400-len(reg_mis_acc))
        reg_mis_acc=extra.extend(reg_mis_acc)
    reg_mis_acc=reg_mis_acc[0:T]
    
    reg_acc=np.array(reg_acc)
    reg_mis_acc=np.array(reg_mis_acc)
    reg_acc=reg_acc.astype(int)
    reg_mis_acc=reg_mis_acc.astype(int)
    
    
    reg_Tcum_acc=np.vstack([reg_Tcum_acc,reg_acc.reshape(1,25400)])                                                     #np.array(reg_T)
    np.savetxt("reg_t1.csv", np.array(reg_Tcum_acc), delimiter=",")                                                      #Our regret on real data
    reg_Tcum_mis_acc=np.vstack([reg_Tcum_mis_acc,reg_mis_acc.reshape(1,25400)])
    np.savetxt("reg_t_acc.csv", np.array(reg_Tcum_mis_acc), delimiter=",")                                   #Regret of RMLP-2 Algorithm on Real Data.



#Algorithm Function using cross-validation via local polynomial to Choose m #
'''
def choose_m(n,v_i_true,y_t):
    array_m=[]
    for m in [2,4,6]:
        pred_1=[]
        pred=[]
        h_range=np.arange(1,5,0.5)
        h_summary=[]
        for h in h_range:
            #h=3
            pred_1=[]
            for i in range(10):
                pred=[]
                #train
                index=np.array(range(i*10,(i+1)*10))
                v_i=np.delete(v_i_true,index)
                y_i=np.delete(y_t,index)
                ker=Kernel(v_i,y_i,n,h*n**(-1/(2*m+1)))
                test_x=v_i_true[(i*10):((i+1)*10)]
                for j in range(10):
                    pred.append(ker.sol_localpoly(4,test_x[j])[0,0])
                    #pred.append(ker.sec_kernel_whole1(test_x[j]))
                pred_1.append(sum((y_t[(i*10):((i+1)*10)]-pred)**2))
            h_summary.append(sum(pred_1))
        array_m.append(min(h_summary))#for every #m, choose the minimum h#
    index_new=np.argmin(array_m) #we choose the minimal $m$ such that the prediction error is minimized.
    if index_new==0: #return m index.
        return 2
    elif index_new==1:
        return 4
    else:
        return 6
'''


# In[ ]:




