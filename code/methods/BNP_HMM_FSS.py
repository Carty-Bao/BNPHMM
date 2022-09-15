# -*- coding: utf-8 -*-
"""
author: Jiadi
Date: 2022.9.8
BNP_HMM_FSS.py for BNPHMM
"""
from gc import callbacks
import sys
sys.path.append('/home/vip/bjd/code')
from hashlib import new
import math
from mimetypes import init
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from synth import dataset_eva, workmode_cat, center_compare, trans_compare, hinton
from scipy.stats import entropy as entropy
from viterbi import viterbiLog
from ChangeFinder import CUSUM_BNP_HMM, ChangeFinder, FSS, CUsum
from HMM_BNP_func import DP_GMM
import time
from multiprocessing import Pool, cpu_count

plt.rc('font',family='Times New Roman')

def BNPHMMFSS(X, Z, batchsize = 20, FSS_threshold=5, unideal_threshold=None):
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
            mean, Z, A = model.del_irr(unideal_threshold)
            minibatch_mu.append(mean)
            minibatch_a.append(A)
            minibatch_z.append(Z)
            initseq = False
            continue
        else:
            model.update(minibatch=minibatch,label = Znibatch,share = True)
            mean, Z, A = model.del_irr(unideal_threshold)
            minibatch_mu.append(mean)
            minibatch_a.append(A)
            minibatch_z.append(Z)
        #如果参差中心不同并且上一个minbatch不是切换点
            if center_compare(minibatch_mu[-1], minibatch_mu[-2],threshold = FSS_threshold)[0] and ~initseq:
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
    return bkps

def add_bkps(bkps):
    global bkps_dic
    bkps_dic.append(bkps)

if __name__ == '__main__':
    

    bkps_dic = []
    bkps_truth_dic = []
    p = Pool(64)
    for i in range(100):
        #读取数据+数据整形
        data_path = "dataset/D7_unideal010/"
        D1 = np.load(data_path+str(i+1)+'.npy', allow_pickle=True)
        #理想数据读取
        # X = np.array([D1[:,0]]).T
        # Z = np.array([D1[:,1]]).T
        # bkps_truth = [150, 300, 450]
        
        #非理想数据读取
        X = np.array([D1[0][:,0]]).T
        Z = np.array([D1[0][:,1]]).T
        bkps_truth = D1[1]

        #切换点检测
        bkps_truth.append(X.shape[0])
        bkps_truth_dic.append(bkps_truth)

        #推理
        # bkps = BNPHMMFSS(X, Z, batchsize = 30, FSS_threshold=3, unideal_threshold=0.01)
        p.apply_async(BNPHMMFSS, args=(X,Z,30,5,0.15,),callback=add_bkps)
        # print(bkps)
        # bkps_dic.append(bkps)
    p.close()
    p.join()
    
    # dataset_eva(bkps_dic, [150, 300, 450, 600])
    #保存
    np.save(data_path + 'result/bkps_dic_BNPHMMFSS.npy',bkps_dic)
    np.save(data_path + 'result/bkps_truth_dic_BNPHMMFSS.npy',bkps_truth_dic)

    dataset_eva(bkps_dic, bkps_truth_dic)


