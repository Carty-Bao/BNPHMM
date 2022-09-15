# -*- coding: utf-8 -*-
"""
author: Jiadi
Date: 2022.5.7
CAVI module for BNP-HMM
"""

from cmath import exp
from email.errors import FirstHeaderLineIsContinuationDefect
import numpy as np
# from main import L
from synth import Non_ideal_HMM, hinton
from scipy.special import digamma, loggamma #gammaln
from hamming import hamming_distance

def inv0(X):#矩阵求逆
    try:
        Xm1 = np.linalg.linalg.inv(X)
        return Xm1
    except IndexError:
        return 1/float(X)

class DP_GMM(object):

    def __init__(self,X,K,Z=None,agile=False):
        self.X = X  #数据(N,XDim)
        self.N, self.XDim = self.X.shape
        self._K = K #截断长度
        self.kappa = 100#跳转系数
        self._hyperparams = {'beta0':(1e-40), #均值的偏差，越小，GMM范围越大
                        'v0':self.XDim+4, #wishart分布的自由度
                        'W0':(1e-2)*np.eye(self.XDim), #协方差的先验，越大，方差越小
                        'alpha':1., #DP-GMM的DP参数
                        'alpha_pi':1.,#初始状态状态的DP参数
                        'alpha_a':35.,#状态转移矩阵的DP参数
                        #DP参数越大，分布越集中
                    }
        self.thre = 1e-4
        self.hamming = []
        if Z is not None:
            self.ztrue = Z#数据标签，用来计算汉明距离
        self.expk = []
        self.agile = agile


    def init_q_param(self):
        #初始化
        self.alpha_pi = self._hyperparams['alpha_pi'] #hyperparam for initial state DP
        self.alpha_a  = self._hyperparams['alpha_a'] #hyperparam for transition DP
        self.alpha = self._hyperparams['alpha'] #hyperparam for DPMM prior


        # self.m0 = np.random.randint(low=np.min(X), high=np.max(X),size=1).astype(float)#采用数据中的某些数作为初始均值，收敛可能会稍快一点
        self.m0 = np.zeros(self.XDim)
        self._W = []
        for k in range(self._K): 
            self._W.append(self._hyperparams['W0']) #初始化协方差矩阵

        self.exp_z = np.array([np.random.dirichlet(np.ones(self._K)) for _ in range(self.N)])#初始化类别矩阵
        self.exp_s = np.array([np.random.uniform(0,100,(self._K,self._K)) for _ in range(self.N)])#初始化状态转移矩阵
        for n in range(self.N): 
            self.exp_s[n,:,:] = self.exp_s[n,:,:] / self.exp_s[n,:,:].sum()#归一化

        self.gamma0, self.gamma1 = np.ones(self._K), np.ones(self._K)#DPMM的BETA分布参数初始化
        self.tau_pi0, self.tau_pi1 = np.ones(self._K), np.ones(self._K)#HMM初始状态分布参数初始化
        self.tau_a0, self.tau_a1 = np.ones((self._K, self._K)), np.ones((self._K, self._K))#HMM状态转移矩阵BETA分布参数初始化

    def init_q_param_online(self):
        '''
        需要保留的变量
        1.self.exp_z
        2.self.exp_s
        3.tau_pi0,tau_pi1
        4.tau_a0,tau_a1
        5.self._m,
        '''
        # self.m0 = np.random.randint(low=np.min(X), high=np.max(X),size=1).astype(float)#采用数据中的某些数作为初始均值，收敛可能会稍快一点
        self.m0 = np.zeros(self.XDim)
        self._W = []
        self.expk = []
        self.hamming = []
        

        for k in range(self._K): 
            self._W.append(self._hyperparams['W0']) #初始化协方差矩阵

        if self.exp_z.shape[0] != self.N:
            self.exp_z = self.exp_z[:self.N]

        
    def mixture_fit(self):
        '''
        DPMM混合模型聚类
        结果是self.exp_z,是类别可能性矩阵，有每个点的类别标签的可能性
        每个集群的均值 self._m
        每个集群的方差 self.expC()
        '''
        itr = 0
        diff,prev_ln_obs_lik = 1,np.zeros((self.N,self._K))
        

        while (itr<5) or (itr<200 and diff>self.thre):           
            #---------------
            # M-step:
            #---------------
            
            self.update_V()#Blei Equ.18 Equ.19
            self.update_m_sigma()
            
            #---------------
            # E-step:
            #---------------
            
            ln_obs_lik = self.loglik()#Bishop Equ 10.64 10.65
            self.ln_obs_lik = ln_obs_lik
            exp_ln_pi = self.eV()#Bishop Equ 10.66
            self.mixEZ(ln_obs_lik, exp_ln_pi) #Bishop Equ 10.46 /Blei Equ 22
            
            diff = abs(ln_obs_lik - prev_ln_obs_lik).sum()/float(self.X.shape[0]*self._K) #average difference in previous expected value of transition matrix
            prev_ln_obs_lik = ln_obs_lik.copy()
            
            # print('itr,diff',itr,diff)

            #next iteration:
            itr+=1
        
            # #calc hanmming distance
            # z_hat = self.exp_z.argmax(axis=1)
            # if self.ztrue is not None:
            #     HD = hamming_distance(self.ztrue,z_hat)
            #     # self.hamming.append(HD)
            
            #calc state number
            del_index = np.where(~self.exp_z.any(axis=0))[0]
            expz = np.delete(self.exp_z, del_index, axis=1)
            _, expk = expz.shape
            self.expk.append(expk)
            


            #determine if we can switch off mix:
            if (itr>=200 or diff<=self.thre) and (itr>=5):
                print ('Mixture converged. SWTCHING TO HMM INFERENCE')
                print ('Mixture Inference itr:', itr)

    def HMM_fit(self):
        '''
        HMM推理
        '''
        itr = 0
        diff,prev_ln_obs_lik = 1,np.zeros((self.N,self._K)) #stop when parameters have converged (local optimum)
        while (itr<3) or (itr<250 and diff>self.thre):            
            #calc state number
            del_index = np.where(~self.exp_z.any(axis=0))[0]
            expz = np.delete(self.exp_z, del_index, axis=1)
            _, expk = expz.shape
            self.expk.append(expk)        
            #---------------
            # M-step:
            #---------------
            
            #variational parameters governing latent states:
            self.mPi()#BJD Equ 1 2
            self.mA()#BJD Equ 3 4
            self.update_m_sigma()#BJD Equ 5-8
            
            #---------------
            # E-step:
            #---------------
            
            ln_obs_lik = self.loglik()#Bishop Equ 10.64
            self.ln_obs_lik = ln_obs_lik
            exp_ln_pi = self.ePi()#Bishop Equ 10.65
            exp_ln_a = self.eA()#Bishop Equ 10.66
            ln_alpha_exp_z = self.eFowardsZ(exp_ln_pi, exp_ln_a, ln_obs_lik) #前向算法
            ln_beta_exp_z = self.eBackwardsZ(exp_ln_pi, exp_ln_a, ln_obs_lik) #后向算法
            self.eZ(ln_alpha_exp_z, ln_beta_exp_z) #前向后向算法 李航 Equ 10.24
            self.eS(exp_ln_a, ln_alpha_exp_z, ln_beta_exp_z, ln_obs_lik) #李航 Equ10.25
            
            diff = abs(ln_obs_lik - prev_ln_obs_lik).sum() / float(self.N*self._K)
            prev_ln_obs_lik = ln_obs_lik.copy()
            
            itr+=1

            #calc hamming distance
            # z_hat = self.exp_z.argmax(axis=1)
            # if self.ztrue is not None:
            #     HD = hamming_distance(self.ztrue, z_hat)
            #     self.hamming.append(HD)

            if (itr>=250 or diff<=self.thre) and (itr>=3):
                print ('HMM INFERENCE done')
                print ('HMM Inference itr:',itr)
        self.exp_pi = self.expPi()#从Beta分布中采样 得到初始状态
        self.exp_a = self.expA()    #从Beta分布中采样 得到状态转移矩阵
    
    def fit(self):
        self.init_q_param()
        self.mixture_fit()
        self.HMM_fit()

    def update(self,minibatch=None,label=None, share=True, add_one = None):
        if add_one is not None:
            #如果采用CUSUM策略，每一次手动向minibatch中加一个脉冲和一个标签
            new_x = np.array([add_one[0]])
            new_z = np.array([add_one[1]])
            self.X = np.append(self.X, new_x, axis=0)
            self.ztrue = np.append(self.ztrue, new_z, axis=0)
            self.N, self.XDim = self.X.shape
            if share:
                self.exp_z = np.append(self.exp_z, np.expand_dims(np.random.dirichlet(np.ones(self._K)),axis=0),axis=0)
                # new_s = np.random.uniform(0,100,(self._K,self._K))
                # new_s[:,:] = new_s[:,:] / new_s[:,:].sum()
                # new_s = np.expand_dims(new_s, axis=0)
                # self.exp_s = np.append(self.exp_s, new_s,axis=0)
                self.exp_s = np.array([np.random.uniform(0,100,(self._K,self._K)) for _ in range(self.N)])#初始化状态转移矩阵
                for n in range(self.N): 
                    self.exp_s[n,:,:] = self.exp_s[n,:,:] / self.exp_s[n,:,:].sum()#归一化
            else:
                self.exp_z = np.array([np.random.dirichlet(np.ones(self._K)) for _ in range(self.N)])#初始化类别矩阵
                self.exp_s = np.array([np.random.uniform(0,100,(self._K,self._K)) for _ in range(self.N)])#初始化状态转移矩阵
                for n in range(self.N): 
                    self.exp_s[n,:,:] = self.exp_s[n,:,:] / self.exp_s[n,:,:].sum()#归一化
                self.mixture_fit()
            self.HMM_fit()
            return

        else:
            assert minibatch is not None, "it is not CUSUM strategy, need to collect a new minibatch"
            assert label is not None, "it is not CUSUM strategy, need to feed labels in Znibatch"
            self.X = minibatch
            self.N, self.XDim = self.X.shape

            if label is not None:
                self.ztrue = label
            if share:
                self.init_q_param_online()
            else:
                self.init_q_param()
            
            # self.mixture_fit()
            self.HMM_fit() 

    def del_irr(self,threshold = None):
        '''
        删除不相关的点，只保留概率较大的点
        thre越大 保留的点越少
        这里thre一般取0.6-1
        '''
        if threshold is None:
            del_index = np.where(~self.exp_z.any(axis=0))[0]
        else:
            allocate = np.argmax(self.exp_z,axis=0)
            del_index=[]
            for i in self._K:
                frequency = allocate.count(i)/



            # states = self.exp_a.sum(axis=0)
            # del_index = [i for i, state in enumerate(states) if state<threshold]

        del_row = np.delete(self.exp_a, del_index, axis=0)
        exp_a = np.delete(del_row, del_index, axis=1)
        _m = np.delete(self._m, del_index)
        exp_z = np.delete(self.exp_z, del_index, axis=1)
        return _m,exp_z,exp_a
    
    def mPi(self):
        #alpha_pi: DP参数
        #exp_z: 类别
        #K: 截断数
        K = self._K
        for k in range(K):
            self.tau_pi0[k] = self.alpha_pi + self.exp_z[0,k+1:].sum() #BJD Equ 2
            self.tau_pi1[k] = 1. + self.exp_z[0,k]  #BJD Equ 1

    def mA(self):
        #alpha_a: 状态转移矩阵DP参数
        #exp_s: 状态转移的类别
        #K: DP截断
        K = self._K
        for i in range(K):
            for j in range(K):
                if i == j and self.agile:
                    self.tau_a0[i,j] = self.alpha_a + self.exp_s[:,i,j+1:].sum()# BJD Equ 4
                    self.tau_a1[i,j] = 1.
                else:
                    self.tau_a0[i,j] = self.alpha_a + self.exp_s[:,i,j+1:].sum() # BJD Equ 4
                    self.tau_a1[i,j] = 1. + self.exp_s[:,i,j].sum()  # BJD Equ 3
                
    def eA(self):
        #Blei Equ 22 前两项
        K = self._K
        exp_ln_a = np.zeros((K,K))
        acc = digamma(self.tau_a0) - digamma(self.tau_a0 + self.tau_a1)
        for i in range(K):
            for j in range(K):
                exp_ln_a[i,j] = digamma(self.tau_a1[i,j]) - digamma(self.tau_a0[i,j] + self.tau_a1[i,j]) + acc[i,:j].sum()
        return exp_ln_a
    
    def eFowardsZ(self,exp_ln_pi,exp_ln_a,ln_obs_lik):
        ln_alpha_exp_z = np.zeros((self.N,self._K)) - np.inf
        #initial state distribution:
        ln_alpha_exp_z[0,:] = exp_ln_pi + ln_obs_lik[0,:]#李航 Equ 10.15
        for n in range(1,self.N):
            for i in range(self._K): #李航 Equ 10.16
                ln_alpha_exp_z[n,:] = np.logaddexp(ln_alpha_exp_z[n,:], ln_alpha_exp_z[n-1,i]+ exp_ln_a[i,:] + ln_obs_lik[n,:])
        return ln_alpha_exp_z 
    
    def eBackwardsZ(self,exp_ln_pi,exp_ln_a,ln_obs_lik):
        N = self.N
        ln_beta_exp_z = np.zeros((self.N,self._K)) - np.inf
        #final state distribution:
        ln_beta_exp_z[N-1,:] = np.zeros(self._K)
        for n in range(N-2,-1,-1):
            for j in range(self._K): #marginalise over all possible next states:
                ln_beta_exp_z[n,:] = np.logaddexp(ln_beta_exp_z[n,:], ln_beta_exp_z[n+1,j] + exp_ln_a[:,j] + ln_obs_lik[n+1,j])
        return ln_beta_exp_z

    def eZ(self, ln_alpha_exp_z, ln_beta_exp_z):
        #李航 Equ 10.24
        ln_exp_z = ln_alpha_exp_z + ln_beta_exp_z
        
        #exponentiate and normalise:
        ln_exp_z -= np.reshape(ln_exp_z.max(axis=1), (self.N,1))
        self.exp_z = np.exp(ln_exp_z) / np.reshape(np.exp(ln_exp_z).sum(axis=1), (self.N,1))

    def eS(self, exp_ln_a, ln_alpha_exp_z, ln_beta_exp_z, ln_obs_lik):
        K = self._K
        N = self.N
        ln_exp_s = np.zeros((N-1,K,K)) #这里不包含初始状态，所以是N-1
        exp_s = np.zeros((N-1,K,K))
        for n in range(N-1):
            for i in range(K):#李航 Equ 10.25
                ln_exp_s[n,i,:] = ln_alpha_exp_z[n,i] + ln_beta_exp_z[n+1,:] + ln_obs_lik[n+1,:]  + exp_ln_a[i,:]
            ln_exp_s[n,:,:] -= ln_exp_s[n,:,:].max()
            exp_s[n,:,:] = np.exp(ln_exp_s[n,:,:]) / np.exp(ln_exp_s[n,:,:]).sum() #归一化 取对数
        self.exp_s = exp_s

    def expPi(self):
        #从Beta分布中采样 得到初始状态
        K = self._K
        exp_pi = np.zeros((1,K))
        acc = self.tau_pi0 / (self.tau_pi0 + self.tau_pi1)
        for k in range(K): 
            exp_pi[0,k] = (acc[:k].prod()*self.tau_pi1[k]) / (self.tau_pi0[k] + self.tau_pi1[k])
        return exp_pi
    
    def expA(self):
        K = self._K
        exp_a = np.zeros((K,K))
        acc = self.tau_a0/(self.tau_a0+self.tau_a1)
        for i in range(K):
            for j in range(K):
                exp_a[i,j] = (acc[i,:j].prod()*self.tau_a1[i,j])/(self.tau_a0[i,j]+self.tau_a1[i,j]) 
        return exp_a

    def update_m_sigma(self):
        (N,XDim) = np.shape(self.X)
        (N1,K) = np.shape(self.exp_z)
        
        v0 = self._hyperparams['v0']
        beta0 = self._hyperparams['beta0'] 
        self._expW0 = self._hyperparams['W0']       

        
        
        NK = self.exp_z.sum(axis=0)#Bishop Equ 10.51
        self._NK = NK
        vk = v0 + NK + 1#Bishop Equ 10.63
        self._vk = vk
        xd, S = self._calc_Xk_Sk()#Bishop Equ 10.52 10.53
        self._xd = xd
        self._S = S
        betak = beta0 + NK#Bishop Equ 10.60
        self._betak = betak
        self._m = self.update_m(K,XDim,beta0)#Bishop Equ 10.61
        self._W = self.update_W(K,XDim,beta0) #Bishop Equ 10.62 

    def loglik(self):
        #计算多元高斯的对数似然，就是说有k个隐变量，对应k个对数似然函数
        K = self._K
        (N,XDim)=np.shape(self.X)
        #数据似然
        exp_diff_mu = self._eDiffMu(XDim,N,K) #eqn 10.64 Bishop
        exp_invc = self._eInvc(XDim, K) #eqn 10.65 Bishop
        ln_lik = 0.5*exp_invc - 0.5*exp_diff_mu

        return ln_lik

    def _eInvc(self,XDim,K):
        invc = [None for _ in range(K)]
        for k in range(K):
            dW = np.linalg.linalg.det(self._W[k])
            if dW > 1e-30: 
                ld = np.log(dW)
            else: ld = 0.0
            invc[k] = sum([digamma((self._vk[k]+1-i) / 2.) for i in range(XDim)]) + XDim * np.log(2) + ld
        return np.array(invc)

    def _eDiffMu(self,XDim,N,K):
        Mu = np.zeros((N,K))
        A = XDim / self._betak
        for k in range(K):
            B0 = (self.X - self._m[k,:]).T
            B1 = np.dot(self._W[k], B0)
            l = (B0*B1).sum(axis=0)
            assert np.shape(l)==(N,),np.shape(l)
            Mu[:,k] = A[k] + self._vk[k]*l 
        
        return Mu

    def _calc_Xk_Sk(self):#Bishop Equ 10.52 10.53
        (N,XDim) = np.shape(self.X)
        (N1,K) = np.shape(self.exp_z)
        assert N==N1
        xd = np.zeros((K,XDim))
        for k in range(K):
            xd[k,:] = (np.reshape(self.exp_z[:,k],(N,1))*self.X).sum(axis=0)
        #safe divide:
        for k in range(K):
            if self._NK[k]>0: xd[k,:] = xd[k,:]/self._NK[k]
        
        S = [np.zeros((XDim,XDim)) for _ in range(K)]
        for k in range(K):
            B0 = np.reshape(self.X - xd[k,:], (N,XDim))
            for d0 in range(XDim):
                for d1 in range(XDim):
                    L = B0[:,d0]*B0[:,d1]
                    S[k][d0,d1] += (self.exp_z[:,k]*L).sum()
        #safe divide:
        for k in range(K):
            if self._NK[k]>0: S[k] = S[k]/self._NK[k]

        return xd, S
    
    def expC(self):
        #calculate expected covariance matrix (for each component)
        return np.array([inv0(Wk*vk) for (Wk,vk) in zip(self._W,self._vk)])
    
    def update_W(self,K,XDim,beta0):#Bishop Equ 10.62
        Winv = [None for _ in range(K)]
        for k in range(K): 
            Winv[k]  = self._NK[k]*self._S[k] + inv0(self._expW0)
            Q0 = np.reshape(self._xd[k,:] - self.m0, (XDim,1))
            q = np.dot(Q0,Q0.T)
            Winv[k] += (beta0 * self._NK[k] / (beta0 + self._NK[k]) ) * q
            assert np.shape(q)==(XDim,XDim)
        W = []
        for k in range(K):
            try:
                W.append(inv0(Winv[k]))
            except np.linalg.linalg.LinAlgError:
                raise np.linalg.linalg.LinAlgError()
        return W
    
    def update_m(self,K,XDim,beta0):#Bishop Equ.10.61
        m = np.zeros((K,XDim))
        for k in range(K): m[k,:] = (beta0*self.m0 + self._NK[k]*self._xd[k,:]) / self._betak[k]
        return m  
    
    def update_V(self):               #Blei Equ.18 Equ.19
    #DP-GMM聚类 首先分清楚属于那个参差集群
        for k in range(self._K):
            self.gamma0[k] = self.alpha + self.exp_z[:,k+1:].sum() #Blei Eqn 19
            self.gamma1[k] = 1. + self.exp_z[:,k].sum() #Blei Eqn 18
    
    def ePi(self):
        #Blei Equ22 前两项
        exp_ln_pi = np.zeros(self._K)
        acc = digamma(self.tau_pi0) - digamma(self.tau_pi0 + self.tau_pi1)
        for k in range(self._K): 
            exp_ln_pi[k] = digamma(self.tau_pi1[k]) - digamma(self.tau_pi0[k] + self.tau_pi1[k]) + acc[:k].sum()
        return exp_ln_pi 

    def eV(self):
        #Blei Equ.22 前两项 
        exp_ln_pi = np.zeros(self._K)
        acc = digamma(self.gamma0) - digamma(self.gamma0 + self.gamma1)
        for k in range(self._K): 
            exp_ln_pi[k] = digamma(self.gamma1[k]) - digamma(self.gamma0[k] + self.gamma1[k]) + acc[:k].sum()
        return exp_ln_pi 

    def mixEZ(self,ln_obs_lik, exp_ln_pi):#Bishop Eqn.10.46
        K = self._K
        N = self.X.shape[0]
        ln_exp_z = np.zeros((N,K))
        for k in range(K):
            ln_exp_z[:,k] = exp_ln_pi[k] + ln_obs_lik[:,k]
        #归一化        
        ln_exp_z -= np.reshape(ln_exp_z.max(axis=1), (N,1))
        self.exp_z = np.exp(ln_exp_z) / np.reshape(np.exp(ln_exp_z).sum(axis=1), (N,1))
    

if __name__ == "__main__":
 
     
    import matplotlib.pyplot as plt
    plt.rc('font',family='Times New Roman')
    N, K_grnd, K = 30, 5, 15
    isagile =  True
    type = 'slide'
    X, Y, Z, mu_grnd = Non_ideal_HMM(N, K_grnd, mu_sd_factor=0.5, spur_ratio=0, miss_ratio=0, type=type)
    plt.scatter(range(len(X)),X)
    plt.show()

    expk_sum = []
    expk_mean = []#估计出的状态个数的平均值
    expk = []
    hd = []
    for itr in range(1):
        print("itr number: ",itr)
        model = DP_GMM(X, K, Z=Z, agile=isagile)
        model.init_q_param()#初始化q分布
        # model.mixture_fit()#DPMM
        model.HMM_fit()
        mu,Z,A = model.del_irr()
        expk_sum.append(model.expk)
        expk.append(len(mu))
        hd.append(model.hamming)
        # del model
    expk_sum = np.array(expk_sum)
    # expk_mean = (np.sum(expk_sum, axis=0)/10)
    print("esitiamted state number: ",np.mean(expk))
    hinton(A)
    print("hamming distance:", hd[0][-1])



    # plt.plot(model.hamming)
    # np.savetxt('results/hamming_not.txt',model.hamming,fmt='%lf')
    # plt.plot(model.expk)
    # plt.plot(expk_mean)
    # np.savetxt('results/statenum '+str(model.agile)+' slide.txt',model.expk,fmt='%d')
    # np.savetxt('results/mean statenum '+str(isagile)+type+'.txt', expk_mean,fmt='%d')



    plt.figure(dpi=150,figsize=(8,5))
    plt.xlabel("PRI index")
    plt.ylabel("PRI value")
    plt.title("slide PRI")



    for i, x in enumerate(X):
        bank = ['r','g','b','black','yellow',"orange",'brown','grey','pink','purple','rosybrown','salmon','silver','tan','wheat']
        z_n = np.argmax(model.exp_z[i])
        plt.plot(i,x,"*-",c=bank[z_n])
    # plt.plot(range(len(X)), X, markersize=6, marker='*')  
    plt.show()