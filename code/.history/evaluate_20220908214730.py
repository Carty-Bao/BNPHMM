import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import m
import seaborn as sns
import pandas as pd
from synth import dataset_eva
plt.rc('font',family='Times New Roman')
sns.set_theme(font='Times New Roman',font_scale=1.4)

if __name__ == '__main__':
    
    bkps_dic = np.load('dataset/D2/result/bkps_dic_UCUSUM.npy', allow_pickle=True)
    bkps_truth = [150,300,450,600]
    FAR, MDR, ADD = dataset_eva(bkps_dic, bkps_truth)   