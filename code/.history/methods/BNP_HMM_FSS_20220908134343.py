# -*- coding: utf-8 -*-
"""
author: Jiadi
Date: 2022.9.8
BNP_HMM_FSS.py for BNPHMM
"""
from hashlib import new
import math
from mimetypes import init
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from synth import workmode_cat, center_compare, trans_compare, hinton, calculate_FAR_MDR_ADD
from scipy.stats import entropy as entropy
from viterbi import viterbiLog
from ChangeFinder import CUSUM_BNP_HMM, ChangeFinder, FSS, CUsum
from HMM_BNP_func import DP_GMM
import time
plt.rc('font',family='Times New Roman')

def BNPHMMFSS(X, Z, bkps_truth, batchsize = 20, FSS_thresholds=[5]):

    N = X.shape[0]
    batch_num = math.ceil(N/batchsize)
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
            mean, Z, A = model.del_irr()
            minibatch_mu.append(mean)
            minibatch_a.append(A)
            minibatch_z.append(Z)
            initseq = False
            continue
        else:
            model.update(minibatch=minibatch,label = Znibatch,share = True)
            mean, Z, A = model.del_irr()
            minibatch_mu.append(mean)
            minibatch_a.append(A)
            minibatch_z.append(Z)
        #如果参差中心不同并且上一个minbatch不是切换点
            if center_compare(minibatch_mu[-1], minibatch_mu[-2],threshold = threshold)[0] and ~initseq:
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

if __name__ == '__main__':

    for i in range(100):
        #读取数据
        data_path = "../dataset/D1/"
        D1 = np.load(data_path+str(i))
        X = np.array([D1[:,0]]).T
        Z = np.array([D1[:,1]]).T

