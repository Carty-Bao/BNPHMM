# -*- coding: utf-8 -*-
"""
author: Jiadi
Date: 2022.5.16
main.py for BNPHMM
"""
#%%
from hashlib import new
import math
from mimetypes import init
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from synth import workmode_cat, center_compare, trans_compare, hinton
from scipy.stats import entropy as entropy
from viterbi import viterbiLog
from ChangeFinder import CUSUM_BNP_HMM, ChangeFinder, FSS, CUsum
from HMM_BNP_func import DP_GMM
import time
plt.rc('font',family='Times New Roman')

#数据导入,修改参数部分
#%%
D1 = np.load('dataset/D8.npy')
X = np.array([D1[:,0]]).T
Z = np.array([D1[:,1]]).T
#数据切割forFSS
#%%
batchsize = 30
N = X.shape[0]
batch_num = math.ceil(N/batchsize)

################################################################    基于FSS算法     ################################################################
################################################################  返回切换点的下标列表  ##############################################################
#%%
#BNP-HMM训练&FSS切换点检测
L = 15#截断长度
minibatch_mu = []#存放每个minibatch的均值
minibatch_a = []#存放每个minibatch的状态转移矩阵
minibatch_z = []
bkps = []
initseq = True#是否是切换点检测任务的开始
start = time.time()
for i in range(batch_num):
    print("minibatch num:", i+1)
    try:
        minibatch = X[i*batchsize:(i+1)*batchsize]
        Znibatch = Z[i*batchsize:(i+1)*batchsize]
    except:
        minibatch = X[i*batchsize:]
        Znibatch = Z[i*batchsize:]
    if initseq==True:
        model = DP_GMM(X = minibatch, K=L,Z=Znibatch,agile=True)
        model.init_q_param()
        model.mixture_fit()
        model.HMM_fit()
        mean, Z, A = model.del_irr(threshold = 0.45)
        minibatch_mu.append(mean)
        minibatch_a.append(A)
        minibatch_z.append(Z)
        initseq = False
        continue
    else:
        model.update(minibatch=minibatch,label = Znibatch,share = True)
        mean, Z, A = model.del_irr(0.45)
        minibatch_mu.append(mean)
        minibatch_a.append(A)
        minibatch_z.append(Z)
    #如果参差中心不同并且上一个minbatch不是切换点
        if center_compare(minibatch_mu[-1], minibatch_mu[-2],threshold = 2)[0] and ~initseq:
            bkps.append(min((i+1)*batchsize, len(X)))
            initseq = True
            del model
            continue

        # if trans_compare(A_1 = minibatch_a[-2], A_2 = minibatch_a[-1], mu_1 = minibatch_mu[-2],mu_2 = minibatch_mu[-1],threshold=5)[0] and ~initseq:
        #     bkps.append(min((i+1)*batchsize, len(X)))
        #     initseq = True
        #     del model
        #     continue
        
print("time consuming:",time.time()-start)
bkps.append(X.shape[0])
print(bkps)
# print(minibatch_mu)

#绘制图片
#%%
# fig, ax = plots(4, 1, figsize=(16, 10),dpi=300)
fig = plt.figure(dpi=300,figsize=(16,8))
plt.xlabel("PRI index")
plt.ylabel("PRI value")
# plt.title("stagger PRI")
bank = ['r','g','b','orange','brown','purple','rosybrown']
plt.plot(range(bkps[0]),X[:bkps[0]],marker="*", color=bank[0])
for index in range(1,len(bkps)):
    plt.plot(range(bkps[index-1]+1, bkps[index]),X[bkps[index-1]+1: bkps[index]], marker="*", color=bank[index])
# plt.savefig("draw/D2_FSS.pdf",dpi=300)

################################################################    基于滑窗的算法     ################################################################
################################################################  返回切换点的下标列表  ##############################################################
#数据切割
#%%
batchsize = 15
N = X.shape[0]

#BNP-HMM参数估计&CUSUM切换点检测
#%%
L = 15#截断长度
minibatch_mu = []#存放每个minibatch的均值
minibatch_a = []#存放每个minibatch的状态转移矩阵
minibatch_z = []
bkps = []
cusum_p = 0
cusum_a = 0
initseq = True#是否是切换点检测任务的开始
start = time.time()
i = 0
while(i<N-batchsize+1):
    print("minibatch num:", i+1)
    try:
        minibatch = X[i:(i+batchsize)]
        Znibatch = Z[i:(i+batchsize)]
    except:
        minibatch = X[i*batchsize:]
        Znibatch = Z[i*batchsize:]
    if initseq==True:
        model = DP_GMM(X = minibatch, K=L,Z=Znibatch,agile=True)
        model.init_q_param()
        model.mixture_fit()
        model.HMM_fit()
        mean, Z, A = model.del_irr() 
        minibatch_mu.append(mean)
        minibatch_a.append(A)
        minibatch_z.append(Z)
        initseq = False
        i+=1
        continue
    else:
        model.update(minibatch=minibatch,label = Znibatch,share = True)
        mean, Z, A = model.del_irr()
        minibatch_mu.append(mean)
        minibatch_a.append(A)
        minibatch_z.append(Z)
        i+=1
    #如果参差中心不同并且上一个minbatch不是切换点
        cusum_p += center_compare(minibatch_mu[-1], minibatch_mu[-2],threshold = 1)[1]
        if cusum_p > 10 and ~initseq:
            bkps.append(min(i+batchsize, len(X)))
            initseq = True
            del model
            cusum_p = 0
            i+=batchsize
            continue

        cusum_a += trans_compare(A_1 = minibatch_a[-2], A_2 = minibatch_a[-1], mu_1 = minibatch_mu[-2],mu_2 = minibatch_mu[-1],threshold=1)[1]
        if cusum_a>1 and ~initseq:
            bkps.append(min(i+batchsize, len(X)))
            initseq = True
            del model
            cusum_a = 0
            i+=batchsize
            continue
        
print("time consuming:",time.time()-start)
bkps.append(N)
print(bkps)

#%%
fig = plt.figure(dpi=300,figsize=(16,8))
plt.xlabel("PRI index")
plt.ylabel("PRI value")
# plt.title("stagger PRI")
bank = ['r','g','b','orange','brown','purple','rosybrown']
plt.plot(range(bkps[0]),X[:bkps[0]],marker="*", color=bank[0])
for index in range(1,len(bkps)):
    plt.plot(range(bkps[index-1]+1, bkps[index]),X[bkps[index-1]+1: bkps[index]], marker="*", color=bank[index])
plt.savefig("draw/D8.pdf",dpi=300)


################################################################    基于Chi2 GLR 的算法     ################################################################
################################################################  返回切换点的下标列表  ##############################################################
# %%
# 数据读取
D1 = np.load('dataset/D7.npy')
X = np.array([D1[:,0]]).T
Z = np.array([D1[:,1]]).T
# 定义基于chi2 GLR的在线切换点检测器
CF_p = CUSUM_BNP_HMM(bkps=[], mean=[], var=[], para_known=False, threshold=1e8)
# 设置初始化的脉冲长度，使用DPMM给个先验
initsize = 40
i = 0
N = X.shape[0]
L = 15          #截断长度
minibatch_mu = []#存放每个minibatch的均值
minibatch_a = []#存放每个minibatch的状态转移矩阵
minibatch_z = []
bkps = []
buffer = []#用来存放当前时刻下所有的数据
initseq = True#是否是切换点检测任务的开始
start = time.time()
while i < N:
    print("pulse num:", i)
    if initseq==True:
        initbatch = X[i:i+initsize]#给出初始化的长度
        Znitbatch = Z[i:i+initsize]
        model = DP_GMM(X = initbatch, K=L,Z=Znitbatch, agile=False)
        model.init_q_param()
        model.mixture_fit()
        model.HMM_fit()
        mean, expZ, expA = model.del_irr()
        scor = CF_p.update(mean)
        minibatch_mu.append(mean)
        init_mu = mean
        minibatch_a.append(expA)
        init_A = expA
        minibatch_z.append(expZ)
        initseq = False
        i += initsize
        continue
    else:
        model.update(share = True,add_one = [X[i],Z[i]])
        mean, expZ, expA = model.del_irr()
        scor = CF_p.update(mean)
        minibatch_mu.append(mean)
        minibatch_a.append(expA)
        minibatch_z.append(expZ)
        i+=1
        if scor > 1 :
            bkps.append(i)
            initseq = True
            del model
            continue
        # print(trans_compare(A_1 = init_A , A_2 = minibatch_a[-1], mu_1 = init_mu,mu_2 = minibatch_mu[-1],threshold=2))
        # if trans_compare(A_1 = init_A , A_2 = minibatch_a[-1], mu_1 = init_mu,mu_2 = minibatch_mu[-1],threshold=0.7)[0] and ~initseq:
        #     bkps.append(i)
        #     initseq = True
        #     del model
        #     continue
bkps.append(N)
print("total time comsuming:" , time.time()-start)
# %%
fig = plt.figure(dpi=300,figsize=(16,8))
plt.xlabel("PRI index")
plt.ylabel("PRI value")
# plt.title("stagger PRI")
bank = ['r','g','b','orange','brown','purple','rosybrown']
plt.plot(range(bkps[0]),X[:bkps[0]],marker="*", color=bank[0])
for index in range(1,len(bkps)):
    plt.plot(range(bkps[index-1]+1, bkps[index]),X[bkps[index-1]+1: bkps[index]], marker="*", color=bank[index])
# plt.savefig("draw/D7.pdf",dpi=300)

################################################################    基于ChangeFinder 的算法     ################################################################
################################################################  返回切换点的下标列表  ##############################################################

#%%
D1 = np.load('dataset/D7.npy')
X = np.array([D1[:,0]]).T
Z = np.array([D1[:,1]]).T
scores = []
bkps = []
rest = 10
start = time.time()
CF = ChangeFinder(r=0.051, order=3, smooth=7, outlier = False)
for index, sig in enumerate(X):
    scor, predict = CF.update(sig)
    rest += 1
    if scor > 0.8 and rest>10:
        bkps.append(index)
        rest = 0
    scores.append(scor)
bkps.append(X.shape[0])
print(bkps)
print("time consuming:",time.time()-start)

#%%
fig = plt.figure(dpi=300,figsize=(16,8))
plt.xlabel("PRI index")
plt.ylabel("PRI value")
# plt.title("stagger PRI")
bank = ['r','g','b','orange','brown','purple','rosybrown']
plt.plot(range(bkps[0]),X[:bkps[0]],marker="*", color=bank[0])
for index in range(1,len(bkps)):
    plt.plot(range(bkps[index-1]+1, bkps[index]),X[bkps[index-1]+1: bkps[index]], marker="*", color=bank[index])
# %%
################################################################    基于U-FSS 的算法     ################################################################
################################################################  返回切换点的下标列表  ##############################################################
D1 = np.load('dataset/D7.npy')
X = np.array([D1[:,0]]).T
Z = np.array([D1[:,1]]).T
start = time.time()
bkps=[]
CF = FSS(X, bkps=[], mean=[], var=[], para_known=False, fixed_threshold=7.8e4, fixed_size=20)#高斯抖动为800
indicater = CF.fss_detection()

for index in range(len(indicater)):
    if indicater[index]>5:
        bkps.append(index)
        rest = 0
bkps.append(X.shape[0])
print(bkps)
print("time consuming:",time.time()-start)
# %%
fig = plt.figure(dpi=300,figsize=(16,8))
plt.xlabel("PRI index")
plt.ylabel("PRI value")
bank = ['r','g','b','orange','brown','purple','rosybrown']
plt.plot(range(bkps[0]),X[:bkps[0]],marker="*", color=bank[0])
for index in range(1,len(bkps)):
    plt.plot(range(bkps[index-1]+1, bkps[index]),X[bkps[index-1]+1: bkps[index]], marker="*", color=bank[index])
# %%
################################################################    基于U-CUSUM 的算法     ################################################################
################################################################  返回切换点的下标列表  ##############################################################
D1 = np.load('dataset/D7.npy')
X = np.array([D1[:,0]]).T
Z = np.array([D1[:,1]]).T
start = time.time()
bkps=[]
CF = CUsum(bkps=[], mean=[], var=[], para_known=False, changepoint_th=1.3e2) 
for index, sig in enumerate(X):
    scor = CF.update(sig)
    if scor>1:
        bkps.append(index)
bkps.append(X.shape[0])
print(bkps)
print("time consuming:",time.time()-start)

#%%
fig = plt.figure(dpi=300,figsize=(16,8))
plt.xlabel("PRI index")
plt.ylabel("PRI value")
bank = ['r','g','b','orange','brown','purple','rosybrown']
plt.plot(range(bkps[0]),X[:bkps[0]],marker="*", color=bank[0])
for index in range(1,len(bkps)):
    plt.plot(range(bkps[index-1]+1, bkps[index]),X[bkps[index-1]+1: bkps[index]], marker="*", color=bank[index])
# %%
