# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from synth import dataset_eva
plt.rc('font',family='Times New Roman')
sns.set_theme(font='Times New Roman',font_scale=1.4)

if __name__ == '__main__':
    
    data_path = 'dataset/D7_unideal020/'
    bkps_dic = np.load(data_path + 'result/bkps_dic_BNPHMMFSS.npy', allow_pickle=True)
    bkps_truth_dic = np.load(data_path + 'result/bkps_truth_dic_BNPHMMFSS.npy', allow_pickle=True)
    # bkps_truth = [150,300,450,600]
    dataset_eva(bkps_dic, bkps_truth_dic)   