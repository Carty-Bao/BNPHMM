# -*- coding: utf-8 -*-
"""
author: Jiadi
Date: 2022.10.2
PE task for experiments
"""

from subprocess import call
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from HMM_BNP_func import DP_GMM
from multiprocessing import Pool
import os
import time
import pdb
plt.rc('font',family='Times New Roman')
sns.set_theme(font='Times New Roman',font_scale=1.4)

def estimate(X, Z, kappa):
    model = DP_GMM(X, K=15, Z=Z, agile=True, kappa = kappa)
    model.init_q_param()
    # model.mixture_fit()
    model.HMM_fit()
    mu, Z, A = model.del_irr(0.03)#估计的非理想比例
    hamming = model.hamming
    del model
    return mu, hamming


data_path = "dataset/D1_unideal005/"
types = ['agile/','dwell/','jitter/','sliding/','stagger/']
# types = ['stagger/']
kappas = np.linspace(0,1,3)
for type in types:
    for kappa in kappas:
        print(type, kappa)
        # global Theta_hat, hammings
        Theta_hat = []
        hammings = []
        res_l = []
        p = Pool(50)
        for i in range(100):
            D = np.load(data_path+type+str(i+1)+'.npy',allow_pickle = True)
            # 理想数据读取
            # X = np.array([D[:,0]]).T
            # Z = np.array([D[:,1]]).T

            # 非理想数据读取
            X = np.array([D[0][:,0]]).T
            Z = np.array([D[0][:,1]]).T
            bkps_truth = D[1]

            # model = DP_GMM(X, K=15, Z=Z, agile=True, kappa = kappa)
            # model.init_q_param()#初始化q分布
            # model.mixture_fit()#DPMM
            # model.HMM_fit()
            res = p.apply_async(estimate, args=(X, Z, kappa,))
            res_l.append(res)

            # mu,Z,A = model.del_irr()
        for res in res_l:
            Theta_hat.append(res.get()[0])
            hammings.append(res.get()[1])
            # del model
        p.close()
        p.join()

        if not os.path.isdir(data_path+type+'result/'):
            os.makedirs(data_path+type+'result/')
        np.save(data_path + type + 'result/Theta_hat_kappa'+str(kappa)+'.npy',Theta_hat)
        np.save(data_path + type + 'result/hamming_kappa'+str(kappa)+'.npy',hammings)