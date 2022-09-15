#%%
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import m
import seaborn as sns
import pandas as pd
plt.rc('font',family='Times New Roman')
sns.set_theme(font='Times New Roman',font_scale=1.4)

#绘制不同调制类型的柱状图
# %%
num = pd.read_csv("../results/stateestimate.csv")
num = pd.DataFrame(num)
num.head(5)

# %%
plt.figure(dpi=300,figsize=(8,5))
fig = sns.barplot(x = 'type', y='deltak', data = num, hue='algorithms')
fig.set_xlabel('')
fig.set_ylabel(r'$\Delta K$')
# %%
barfig = fig.get_figure()
barfig.savefig("../results/estimatedstate.png",dpi=300)


#绘制agile和非agile的在缺失脉冲影响下的折线图
#%% 
missing_num = pd.read_csv('../results/missing.csv')
missing_num = pd.DataFrame(missing_num)
missing_num['deltak'] = missing_num['deltak'].abs()
missing_num.head(5)
#%%
def plot_missing(df, flds):
   # CREATE NEW COLUMN OF CONCATENATED VALUES
   df['_'.join(flds)] =  pd.Series(df.reindex(flds, axis='columns')
                                     .astype('str')
                                     .values.tolist()
                                  ).str.join('_')

   # PLOT WITH hue
   plt.figure(dpi=300,figsize=(10,8))
   fig = sns.lineplot(x='missing pulses', y='deltak', hue='_'.join(flds), data=missing_num)
   plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1))
   plt.ylabel(r'$\Delta K$')
   linefig = fig.get_figure()
   linefig.savefig("../results/estimatedstate_missing.png",dpi=300, bbox_inches = 'tight')
# %%
plot_missing(missing_num, ['algorithms', 'type'])


# 绘制agile和非agile的在虚假脉冲影响下的折线图
# %%
spur_num = pd.read_csv('../results/spur.csv')
spur_num = pd.DataFrame(spur_num)
spur_num['deltak'] = spur_num['deltak'].abs()
spur_num.head(5)

#%%
def plot_spur(df, flds):
   # CREATE NEW COLUMN OF CONCATENATED VALUES
   df['_'.join(flds)] =  pd.Series(df.reindex(flds, axis='columns')
                                     .astype('str')
                                     .values.tolist()
                                  ).str.join('_')

   # PLOT WITH hue
   plt.figure(dpi=300,figsize=(10,8))
   fig = sns.lineplot(x='spur pulses', y='deltak', hue='_'.join(flds), data=spur_num)
   plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1))
   plt.ylabel(r'$\Delta K$')
   linefig = fig.get_figure()
   linefig.savefig("../results/estimatedstate_spur.png",dpi=300, bbox_inches = 'tight')
# %%
plot_spur(spur_num, ['algorithms', 'type'])

#绘制不同调制类型下的汉明距离
# %%
missing_hammming = pd.read_csv('../results/hamming_missing.csv')
missing_hammming = pd.DataFrame(missing_hammming)
missing_hammming.head(5)

#%%
def plot_spur(df, flds):
   # CREATE NEW COLUMN OF CONCATENATED VALUES
   df['_'.join(flds)] =  pd.Series(df.reindex(flds, axis='columns')
                                     .astype('str')
                                     .values.tolist()
                                  ).str.join('_')

   # PLOT WITH hue
   plt.figure(dpi=300,figsize=(10,8))
   fig = sns.lineplot(x='missing pulses', y='hamming distance', hue='_'.join(flds), data=missing_hammming)
   plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
   plt.ylabel("hamming distance")
   linefig = fig.get_figure()
   linefig.savefig("../results/hamming_missing.png",dpi=300, bbox_inches = 'tight')
# %%
plot_spur(missing_hammming, ['type'])
# %%
spur_hamming = pd.read_csv('../results/hamming spur.csv')
spur_hamming = pd.DataFrame(spur_hamming)
spur_hamming.head(5)

#%%
def plot_spur(df, flds):
   # CREATE NEW COLUMN OF CONCATENATED VALUES
   df['_'.join(flds)] =  pd.Series(df.reindex(flds, axis='columns')
                                     .astype('str')
                                     .values.tolist()
                                  ).str.join('_')

   # PLOT WITH hue
   plt.figure(dpi=300,figsize=(10,8))
   fig = sns.lineplot(x='spurious pulses', y='hamming distance', hue='_'.join(flds), data=spur_hamming)
   plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
   plt.ylabel("hamming distance")
   linefig = fig.get_figure()
   linefig.savefig("../results/spur_hamming.png",dpi=300, bbox_inches = 'tight')
# %%
plot_spur(spur_hamming, ['type'])

#绘制切换点检测的结果
# %%
def display_signal_scores(signal, bkps):
   alarm = []
   alarm.append(bkps[0])
   for i in range(len(bkps)-1):
      alarm.append(bkps[i+1]-bkps[i])
      fig, ax = plt.subplots(1, 1, figsize=(17, 10))

   for index, bkp in enumerate(alarm):
      ax[0].plot(range(bkp*index, bkp*(index+1)), signal[bkp*index, bkp*(index+1)], marker="*", color=index)

   ax[0].set_ylabel("PRI value", fontsize=18)
   ax[0].set_title("stagger PRI", fontsize=18)
   ax[0].set_xticks([])
   ax[0].tick_params(axis = 'y', which = 'major', labelsize = 15)
   # axins = inset_axes(ax[0], width="50%", height="25%", loc='upper left',
   #                bbox_to_anchor=(0.25, 0.07, 0.8, 0.8),
   #                bbox_transform=ax[0].transAxes)
   # axins.scatter(range(len(signal)), signal, marker="+", color='b')
   # axins.set_xlim(200, 250)
   # axins.set_ylim(85, 97)
   # mark_inset(ax[0], axins, loc1=3, loc2=4, fc="none", ec='r', lw=1, linestyle='--')

   # ax[1].plot(score, label = 'BNP-HMM')
   # ax[1].set_xlabel("PRI index/s", fontsize=12)
   # ax[1].set_ylabel('score', fontsize=18)    
   # ax[1].legend(loc='best',prop={'family' : 'Times New Roman', 'size'   : 15})
   # ax[1].tick_params(axis = 'y', which = 'major', labelsize = 15)

   plt.show()

#%%
D1 = np.load('../dataset/D1.npy')
X = np.array([D1[:,0]]).T
Z = np.array([D1[:,1]]).T
bkps = [330,660]
display_signal_scores(X, bkps)

# %%
num = pd.read_csv("../curves/FSS.csv",header=None).T
num = pd.DataFrame(num)
num.columns = ['FAR', 'ADD']
num.head(5)

# %%
plt.figure(dpi=300,figsize=(8,5))
fig = sns.lineplot(x = 'FAR', y='ADD', data = num)
fig.set_xlabel('False Alarm Rate')
fig.set_ylabel('Average Detection Delay')
# %%
barfig = fig.get_figure()
barfig.savefig("../results/estimatedstate.png",dpi=300)

#%% 
num = np.load('../dataset/D2/bkps_dic.npy', allow_pickle=True)

# %%
