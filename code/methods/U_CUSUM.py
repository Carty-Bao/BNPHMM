# -*- coding: utf-8 -*-
"""
author: Jiadi
Date: 2022.9.8
BNP_HMM_CUSUM.py for BNPHMM
"""
import sys
sys.path.append('/home/vip/bjd/code')
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

def UCUSUM(X, Z, CUSUM_threshold):

    start = time.time()
    bkps=[]
    CF = CUsum(bkps=[], mean=[], var=[], para_known=False, changepoint_th=CUSUM_threshold) 
    for index, sig in enumerate(X):
        scor = CF.update(sig)
        if scor>1:
            bkps.append(index)
    bkps.append(X.shape[0])
    print("time consuming:",time.time()-start)
    return bkps

if __name__ == '__main__':
    
    bkps_dic = []
    bkps_truth_dic = []
    for i in range(100):
        #读取数据+数据整形
        data_path = "dataset/D7/"
        D1 = np.load(data_path+str(i+1)+'.npy',allow_pickle=True)
        #理想数据读取
        X = np.array([D1[:,0]]).T
        Z = np.array([D1[:,1]]).T
        bkps_truth = [150, 300, 450]
        
        #非理想数据读取
        # X = np.array([D1[0][:,0]]).T
        # Z = np.array([D1[0][:,1]]).T
        # bkps_truth = D1[1]

        #切换点检测
        bkps_truth.append(X.shape[0])
        bkps_truth_dic.append(bkps_truth)
        #推理
        bkps = UCUSUM(X, Z, CUSUM_threshold=180)
        print(bkps)
        bkps_dic.append(bkps)

    #保存
    np.save(data_path+'result/bkps_dic_UCUSUM.npy',bkps_dic)
    np.save(data_path+'result/bkps_truth_dic_UCUSUM.npy',bkps_truth_dic)
    dataset_eva(bkps_dic, bkps_truth_dic)