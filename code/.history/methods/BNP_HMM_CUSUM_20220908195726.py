# -*- coding: utf-8 -*-
"""
author: Jiadi
Date: 2022.9.8
BNP_HMM_CUSUM.py for BNPHMM
"""
import sys
sys.path.append('/Users/cartybao/Desktop/2021/HMM建模切换点和分选/code/')
from hashlib import new
import math
from mimetypes import init
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from synth import workmode_cat, center_compare, trans_compare, hinton, dataset_eva
from scipy.stats import entropy as entropy
from viterbi import viterbiLog
from ChangeFinder import CUSUM_BNP_HMM, ChangeFinder, FSS, CUsum
from HMM_BNP_func import DP_GMM
import time
plt.rc('font',family='Times New Roman')

def BNPHMMCUSUM(X, Z, CUSUM_threshold = 10, unideal_threshold = None):

    # 定义基于chi2 GLR的在线切换点检测器
    CF_p = CUSUM_BNP_HMM(bkps=[], mean=[], var=[], para_known=False, threshold=CUSUM_threshold)
    # 设置初始化的脉冲长度，使用DPMM给个先验
    initsize = 50
    i = 0
    N = X.shape[0]
    L = 15          #截断长度
    minibatch_mu = []#存放每个minibatch的均值
    minibatch_a = []#存放每个minibatch的状态转移矩阵
    minibatch_z = []
    bkps = []
    initseq = True#是否是切换点检测任务的开始
    start = time.time()
    while i < N:
        # print("pulse num:", i)
        if initseq==True:
            initbatch = X[i:i+initsize]#给出初始化的长度
            Znitbatch = Z[i:i+initsize]
            model = DP_GMM(X = initbatch, K=L,Z=Znitbatch, agile=False)
            model.init_q_param()
            model.mixture_fit()
            model.HMM_fit()
            mean, expZ, expA = model.del_irr(unideal_threshold)
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
            mean, expZ, expA = model.del_irr(unideal_threshold)
            scor = CF_p.update(mean)
            minibatch_mu.append(mean)
            minibatch_a.append(expA)
            minibatch_z.append(expZ)
            i+=1
            if scor > 1 :
                print("change point detected: ",i)
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
    return bkps

if __name__ == '__main__':
    from multiprocessing import Pool, cpu_count
    import os
    import time
    
    bkps_dic = []
    test_path = "dataset/D2/"
    D1 = np.load(test_path+str(1)+'.npy')
    X = np.array([D1[:,0]]).T
    Z = np.array([D1[:,1]]).T

    # bkps = BNPHMMCUSUM(X, Z, CUSUM_threshold=7, unideal_threshold=None)
    print("CPU内核数:{}".format(cpu_count()))
    print('当前母进程: {}'.format(os.getpid()))
    start = time.time()
    p = Pool(4)
    for i in range(5):
        p.apply_async(BNPHMMCUSUM, args=(X,Z,7,None,))
    print('等待所有子进程完成。')
    p.close()
    p.join()

    print(bkps)
    bkps_dic.append(bkps)
    dataset_eva(bkps_dic, bkps_truth=[150,300,450,600])

    # bkps_dic = []
    # for i in range(100):
    #     #读取数据+数据整形
    #     data_path = "dataset/D2/"
    #     D1 = np.load(data_path+str(i+1)+'.npy')
    #     X = np.array([D1[:,0]]).T
    #     Z = np.array([D1[:,1]]).T
    #     #推理
    #     bkps = BNPHMMCUSUM(X, Z, CUSUM_threshold=7, unideal_threshold=None)
    #     bkps_dic.append(bkps)
    # bkps_dic = np.array(bkps_dic)
    # #保存
    # np.save('dataset/D2/result/bkps_dic_BNPHMMCUSUM.npy',bkps_dic)


