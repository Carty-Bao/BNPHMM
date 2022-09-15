# -*- coding: utf-8 -*-
"""
author: Jiadi
Date: 2022.9.8
BNP_HMM_CUSUM.py for BNPHMM
In order to draw a false alarm and ADD curves.
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
from multiprocessing import Pool, cpu_count
import os
plt.rc('font',family='Times New Roman')
sys.path.append('/home/vip/bjd/code/methods')
from U_CUSUM import UCUSUM

def add_bkps(bkps):
    global bkps_dic
    print(bkps)
    bkps_dic.append(bkps)
if __name__ == '__main__':
    bkps_dic = []
    bkps_truth_dic = []
    thresholds = np.linspace(100,200,50)

    for th_index, th in enumerate(thresholds):
        p = Pool(64)
        for i in range(100):
            #读取数据+数据整形
            data_path = "dataset/D2/"
            D1 = np.load(data_path+str(i+1)+'.npy',allow_pickle = True)
            X = np.array([D1[:,0]]).T
            Z = np.array([D1[:,1]]).T
            # bkps_truth = D1[1]
            bkps_truth = [150,300,450]
            bkps_truth.append(X.shape[0])
            bkps_truth_dic.append(bkps_truth)
            #推理
            p.apply_async(UCUSUM, args=(X,Z,th,), callback=add_bkps)
        p.close()
        p.join()
            # bkps = UCUSUM(X, Z, CUSUM_threshold=th)
            # print(bkps)
            # bkps_dic.append(bkps)
        np.save('curves/D2_UCUSUM/bkps_dic_'+str(th_index)+'.npy',bkps_dic)
        np.save('curves/D2_UCUSUM/bkps_truth_dic_'+str(th_index)+'.npy',bkps_truth_dic)
            #保存
        # dataset_eva(bkps_dic, bkps_truth_dic)

