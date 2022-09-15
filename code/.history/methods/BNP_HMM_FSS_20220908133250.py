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

def 

if __name__ == '__main__':

    #读取数据
    data_path = "../dataset/D1/"
    D1 = np.load('../dataset/D7_unideal.npy')
    X = np.array([D1[:,0]]).T
    Z = np.array([D1[:,1]]).T

