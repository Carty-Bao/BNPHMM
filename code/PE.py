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
import time
plt.rc('font',family='Times New Roman')
sns.set_theme(font='Times New Roman',font_scale=1.4)

def estimate(model):
    model.fit()
    mu, Z, A = model.del_irr()
    hamming = model.hamming
    return mu, hamming

def add_Theta_hamming(mu, hamming):
    global Theta_hat, hammings
    Theta_hat.append(mu)
    hammings.append(hamming)

if __name__ == "__main__":
    data_path = "dataset/D1/"
    types = ['agile/','dwell/','jitter/','sliding/','stagger/']

    kappas = np.linspace(0,1,5)
    for type in types:
        for kappa in kappas:
            print(type, kappa)
            Theta_hat = []
            hammings = []
            # p = Pool(50)
            for i in range(100):
                D = np.load(data_path+type+str(i+1)+'.npy',allow_pickle = True)
                X = np.array([D[:,0]]).T
                Z = np.array([D[:,1]]).T
                model = DP_GMM(X, K=15, Z=Z, agile=True, kappa = kappa)
                model.init_q_param()#初始化q分布
                model.mixture_fit()#DPMM
                model.HMM_fit()
                # p.apply_async(estimate, args=(model, ), callback=add_Theta_hamming)
                mu,Z,A = model.del_irr()
                Theta_hat.append(mu)
                hammings.append(model.hamming)
            np.save(data_path + type + 'result/Theta_hat_kappa'+str(kappa)+'.npy',Theta_hat)
            np.save(data_path + type + 'result/hamming_kappa'+str(kappa)+'.npy',hammings)