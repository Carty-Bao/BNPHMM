# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from synth import dataset_eva
plt.rc('font',family='Times New Roman')
sns.set_theme(font='Times New Roman',font_scale=1.4)

if __name__ == '__main__':
    
    # data_path = 'dataset/D7_unideal020/'
    # bkps_dic = np.load(data_path + 'result/bkps_dic_BNPHMMFSS.npy', allow_pickle=True)
    # bkps_truth_dic = np.load(data_path + 'result/bkps_truth_dic_BNPHMMFSS.npy', allow_pickle=True)
    i=1
    FARs_BNPHMMC, ADDs_BNPHMMC, MT2FAs_BNPHMMC = [], [], []
    for i in range(10):
        bkps_dic = np.load('curves/D2_BNPHMMC/bkps_dic_'+str(i)+'.npy',allow_pickle=True)
        bkps_truth_dic = np.load('curves/D2_BNPHMMC/bkps_truth_dic_'+str(i)+'.npy',allow_pickle=True)
        print(bkps_dic)
        # dataset_eva(bkps_dic, bkps_truth_dic)   
        FAR, MDR, ADD, F1, MT2FA = dataset_eva(bkps_dic, bkps_truth_dic)
        FARs_BNPHMMC.append(FAR)
        ADDs_BNPHMMC.append(ADD)
        MT2FAs_BNPHMMC.append(MT2FA)