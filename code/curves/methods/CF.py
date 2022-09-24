# -*- coding: utf-8 -*-
"""
author: Jiadi
Date: 2022.9.22
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
from Change_finder import Change_Finder

def add_bkps(bkps):
    global bkps_dic
    bkps_dic.append(bkps)

if __name__ == '__main__':

    rs = np.linspace(0,0.5,20)
    orders = [1,2,3,4]

    for order_index, order in enumerate(orders):
        for r_index, r in enumerate(rs):
            bkps_dic = []
            bkps_truth_dic = []
            p = Pool(50)
            for i in range(50):
                #读取数据+数据整形
                data_path = "dataset/D7/"
                D1 = np.load(data_path+str(i+1)+'.npy', allow_pickle=True)
                
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
                # bkps = Change_Finder(X, Z, r=0.1, order=1, smooth=7, outlier= False, CF_threshold = 0.85)
                p.apply_async(Change_Finder, args=(X,Z,r,order,7, False, 0.5,), callback=add_bkps)
                # bkps_dic.append(bkps)
            p.close()
            p.join()
            #保存
            folder_name = data_path[-3:-1] + '_CF/'
            np.save('curves/' + folder_name + 'bkps_dic_'+str(r_index)+'_'+str(order_index)+'.npy',bkps_dic)
            np.save('curves/' + folder_name + 'bkps_truth_dic_'+str(r_index)+'_'+str(order_index)+'.npy',bkps_truth_dic)
            # dataset_eva(bkps_dic, bkps_truth_dic)
