# -*- coding: utf-8 -*-
"""
author: Jiadi
Date: 2022.5.7
CAVI module for BNP-HMM

All numbered ELBO terms & parameter update functions of q refer to respective 
equations in my note "VI for DPMM"
"""

import numpy as np
from synth import Non_ideal_HMM
from scipy.special import digamma, loggamma #gammaln

def inv0(X):#inverts X if it is a matrix, otherwise, it finds numerical inverse
    try:
        Xm1 = np.linalg.linalg.inv(X)
        return Xm1
    except IndexError:
        return 1/float(X)

class DP_GMM(object):

    def __init__(self,X, Y, K,XDim):
        self._K = K #截断长度
        self.X = X
        self.Y = Y
        self._hyperparams = {'beta0':(1e-20), #variance of mean (smaller: broader the means)
                            'v0':XDim+1, #degrees of freedom in inverse wishart
                            'm0':np.zeros(XDim), #prior mean
                            'W0':(1e0)*np.eye(XDim), #prior cov (bigger: smaller covariance)
                            'alpha_c':1.,
                            'alpha_pi':1.0,#hyperparam for initial state DP
                            'alpha_a':1.,#hyperparam for transition DP
                        }
        self._W = []
        for k in range(K): 
            self._W.append(self._hyperparams['W0']) #初始化协方差矩阵 

    def randInit(self,K):
        0

    def fit(self):
        

        self.alpha_pi = self._hyperparams['alpha_pi'] #hyperparam for initial state DP
        self.alpha_a  = self._hyperparams['alpha_a'] #hyperparam for transition DP
        self.alpha_c = self._hyperparams['alpha_c']
        
        self.exp_z = np.array([np.random.dirichlet(np.ones(K)) for _ in range(N)])

        itr = 0
        diff, prev_ln_obs_lik = 1,np.zeros((N,K)) #stop when parameters have converged (local optimum)
        thre = 1e-2
        while (itr<10) or (itr<200 and diff>thre):           
            #---------------
            # M-step:
            #---------------
            
            #variational parameters governing latent states:
            tau_pi0,tau_pi1 = self.update_pi()#Blei Equ.18 Equ.19
            self.update_m_sigma()
            
            #---------------
            # E-step:
            #---------------
            
            #calculate observation likelihood of data for each sensor (combined):
            ln_obs_lik = self.loglik()
            exp_ln_pi = self.ePi(tau_pi0,tau_pi1)
            #find expected values of latent variables:
            self.exp_z = self.mixEZ(ln_obs_lik, exp_ln_pi) #mixture model estimation of Z
            
            
            
            diff = abs(ln_obs_lik - prev_ln_obs_lik).sum()/float(N*K) #average difference in previous expected value of transition matrix
            prev_ln_obs_lik = ln_obs_lik.copy()
            
            print('itr,diff',itr,diff)

            #next iteration:
            itr+=1
            
            #determine if we can switch off mix:
            if (itr>=200 or diff<=thre) and (itr>=10):
                print ('Mixture converged. SWTCHING TO HMM INFERENCE')

    def update_m_sigma(self,randInit=0):
        #optimise variational parameters:
        
        (N,XDim) = np.shape(self.X)
        (N1,K) = np.shape(self.exp_z)
        
        #access hyperparameters
        v0 = self._hyperparams['v0']
        beta0 = self._hyperparams['beta0']
        m0 = self._hyperparams['m0']
        self._expW0 = self._hyperparams['W0']
        
        
        NK = self.exp_z.sum(axis=0)
        vk = v0 + NK + 1
        xd = self._mXd(self.exp_z,self.X)
        S = self._mS(self.exp_z,self.X,xd,NK)
        betak = beta0 + NK
        self._m = self._mM(K,XDim,beta0,m0,NK,xd,betak)
        self._W = self._mW(K,self._expW0,xd,NK,m0,XDim,beta0,S) 
        self._xd = xd
        self._S = S
        self._NK = NK
        self._vk = vk
        self._betak = betak
        tau_ctk = self.alpha_c + np.dot(self.Y.T, self.exp_z)
        self._exp_ln_ctk = digamma(tau_ctk) - digamma(tau_ctk.sum(axis=0))

    def loglik(self):
        #计算多元高斯的对数似然，就是说有k个隐变量，对应k个对数似然函数
        #return log liklihood of each data point x latent component
        K = self._K
        (N,XDim)=np.shape(self.X)
        #calculate some features of the data:
        exp_diff_mu = self._eDiffMu(self.X,XDim,self._NK,self._betak,self._m,self._W,self._xd,self._vk,N,K) #eqn 10.64 Bishop,计算对数似然
        exp_invc = self._eInvc(self._W,self._vk,XDim,K) #eqn 10.65 Bishop
        ln_lik = 0.5*exp_invc - 0.5*exp_diff_mu

        prior_lik = np.dot(self.Y,self._exp_ln_ctk)
        return np.array([ln_lik,prior_lik]).sum(axis=0)

    def _eInvc(self,W,vk,XDim,K):
        invc = [None for _ in range(K)]
        for k in range(K):
            dW = np.linalg.linalg.det(W[k])
            if dW > 1e-30: 
                ld = np.log(dW)
            else: ld = 0.0
            invc[k] = sum([digamma((vk[k]+1-i) / 2.) for i in range(XDim)]) + XDim * np.log(2) + ld
        return np.array(invc)

    def _eDiffMu(self,X,XDim,NK,betak,m,W,xd,vk,N,K):
        Mu = np.zeros((N,K))
        A = XDim / betak
        for k in range(K):
            B0 = (X - m[k,:]).T
            B1 = np.dot(W[k], B0)
            l = (B0*B1).sum(axis=0)
            assert np.shape(l)==(N,),np.shape(l)
            Mu[:,k] = A[k] + vk[k]*l 
        
        return Mu

    def _mXd(self,Z,X):
        #weighted means (by component responsibilites)
        (N,XDim) = np.shape(X)
        (N1,K) = np.shape(Z)
        NK = Z.sum(axis=0)
        assert N==N1
        xd = np.zeros((K,XDim))
        for k in range(K):
            xd[k,:] = (np.reshape(Z[:,k],(N,1))*X).sum(axis=0)
        #safe divide:
        for k in range(K):
            if NK[k]>0: xd[k,:] = xd[k,:]/NK[k]
        
        return xd
    
    def _mS(self,Z,X,xd,NK):
        (N,K)=np.shape(Z)
        (N1,XDim)=np.shape(X)
        assert N==N1
        
        S = [np.zeros((XDim,XDim)) for _ in range(K)]
        for k in range(K):
            B0 = np.reshape(X - xd[k,:], (N,XDim))
            for d0 in range(XDim):
                for d1 in range(XDim):
                    L = B0[:,d0]*B0[:,d1]
                    S[k][d0,d1] += (Z[:,k]*L).sum()
        #safe divide:
        for k in range(K):
            if NK[k]>0: S[k] = S[k]/NK[k]
        return S
    
    def expC(self):
        #calculate expected covariance matrix (for each component)
        return np.array([inv0(Wk*vk) for (Wk,vk) in zip(self._W,self._vk)])
    
    def _mW(self,K,W0,xd,NK,m0,XDim,beta0,S):
        Winv = [None for _ in range(K)]
        for k in range(K): 
            Winv[k]  = NK[k]*S[k] + inv0(W0)
            Q0 = np.reshape(xd[k,:] - m0, (XDim,1))
            q = np.dot(Q0,Q0.T)
            Winv[k] += (beta0*NK[k] / (beta0 + NK[k]) ) * q
            assert np.shape(q)==(XDim,XDim)
        W = []
        for k in range(K):
            try:
                W.append(inv0(Winv[k]))
            except np.linalg.linalg.LinAlgError:
                #print 'Winv[%i]'%k, Winv[k]
                raise np.linalg.linalg.LinAlgError()
        return W
    
    def _mM(self,K,XDim,beta0,m0,NK,xd,betak):
        m = np.zeros((K,XDim))
        for k in range(K): m[k,:] = (beta0*m0 + NK[k]*xd[k,:]) / betak[k]
        return m  

    def update_pi(self):               #Blei Equ.18 Equ.19
        #DP-GMM聚类 首先分清楚属于那个参差集群
        #alpha_pi: hyperparam for DP prior
        #self.exp_z: expectation of latent variables (we are only interested at time step 0 here)
        #K: truncation param. for DP
        K = self._K
        tau_pi0,tau_pi1 = np.zeros(K), np.zeros(K)
        for k in range(K):
            tau_pi0[k] = self.alpha_pi + self.exp_z[:,k+1:].sum() #hyperparam for this component NOT explaining the data
            tau_pi1[k] = 1. + self.exp_z[:,k].sum() #hyperparam for this component explaining the data
        return tau_pi0, tau_pi1
    
    def ePi(self,tau_pi0,tau_pi1):
        K = self._K
        exp_ln_pi = np.zeros(K)
        acc = digamma(tau_pi0) - digamma(tau_pi0 + tau_pi1)
        for k in range(K): 
            exp_ln_pi[k] = digamma(tau_pi1[k]) - digamma(tau_pi0[k] + tau_pi1[k]) + acc[:k].sum()
        return exp_ln_pi      

    def mixEZ(self, ln_obs_lik, exp_ln_pi):#Blei Equ 22
    #follow mixture (not a time series):
        N = self.X.shape[0]
        ln_exp_z = np.zeros((N,K))
        for k in range(K):
            ln_exp_z[:,k] = exp_ln_pi[k] + ln_obs_lik[:,k]

        #exponentiate and normalise:
        ln_exp_z -= np.reshape(ln_exp_z.max(axis=1), (N,1))
        exp_z = np.exp(ln_exp_z) / np.reshape(np.exp(ln_exp_z).sum(axis=1), (N,1))
        return exp_z  
# class BNP_HMM(object):



if __name__ == "__main__":
 
    N, K_grnd, K = 100, 5, 8
    X, Y, mu_grnd = Non_ideal_HMM(N, K_grnd, mu_sd_factor=0.1, spur_ratio=0, miss_ratio=0,type='slide')
    (N,XDim) = np.shape(X)


    model = DP_GMM(X, Y, K, XDim)
    model.fit()
    

    # print(self.exp_z)
    exp_mu, exp_C = model._m, model.expC()
    print(exp_mu)

    import matplotlib.pyplot as plt
    plt.rc('font',family='Times New Roman')

    plt.figure(dpi=150,figsize=(8,5))
    plt.xlabel("PRI index")
    plt.ylabel("PRI value")
    plt.title("dwelling PRI")

    for i, x in enumerate(X):
        bank = ['r','g','b','black','yellow',"orange",'brown','grey','pink','purple']
        z_n = np.argmax(model.exp_z[i])
        plt.scatter(i,x,marker="*",c=bank[z_n])
    # plt.plot(range(len(X)), X, markersize=6, marker='*')  
    plt.show()