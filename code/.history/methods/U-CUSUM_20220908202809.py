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

def UCUSUM(X, Z, FSS_threshold = 5):

    start = time.time()
    bkps=[]
    CF = FSS(X, bkps=[], mean=[], var=[], para_known=False, fixed_threshold=FSS_threshold, fixed_size=batchsize)#高斯抖动为800
    indicater = CF.fss_detection()

    for index in range(len(indicater)):
        if indicater[index]>5:
            bkps.append(index)
    bkps.append(X.shape[0])
    print("time consuming:",time.time()-start)
    return bkps