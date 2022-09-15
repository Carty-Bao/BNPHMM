#%%
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import m
import seaborn as sns
import pandas as pd
import synth
plt.rc('font',family='Times New Roman')
sns.set_theme(font='Times New Roman',font_scale=1.4)
#数据生成
# %%
N, K_grnd, K = 100, 5, 10
X1, _, Z, mu_grd = Non_ideal_HMM(N, K_grnd, mu_sd_factor= 0, spur_ratio=0., miss_ratio=0,type='dwell')