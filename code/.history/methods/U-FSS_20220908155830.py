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

def U-FSS(X, Z, batchsize = 20, FSS_threshold = 5):

    start = time.time()
    bkps=[]
    CF = FSS(X, bkps=[], mean=[], var=[], para_known=False, fixed_threshold=7.8e4, fixed_size=20)#高斯抖动为800
    indicater = CF.fss_detection()

    for index in range(len(indicater)):
        if indicater[index]>5:
            bkps.append(index)
    bkps.append(X.shape[0])
    print("time consuming:",time.time()-start)