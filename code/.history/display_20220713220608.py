# -*- coding: UTF-8 -*-
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
font_title = {  # 用 dict 单独指定 title 样式
    'family': 'Times New Roman',
    'weight': '600',
    'size': 15,
    'usetex' : True,
}

def display_signalscores(signal, signal_pure=None, score=None, origin_score=None, mode='norm'):

    if mode == 'PRI':
        plt.subplot(211)
        plt.scatter(range(len(signal)), signal, marker='+', color='b')
        plt.scatter(range(len(signal_pure)), signal_pure, marker='*',color='r')
        # plt.ylim(min(signal)-10, max(signal)+10)
        plt.ylim(0, 200)
        plt.title('signal')
        plt.xlabel('sequence')
        plt.ylabel('amplitude')

        if score is not None:
            plt.subplot(212)
            plt.scatter(range(len(score)),score)
            plt.title('score')
            plt.xlabel('sequence')
            plt.ylabel('score')
        plt.show()

    if mode == 'norm':
        plt.subplot(311)
        plt.plot(signal)
        plt.title('高斯抖动PRI脉冲',fontproperties="Songti SC", fontsize=18)
        # plt.xlabel('脉冲索引',fontproperties="Songti SC", fontsize=18)
        plt.ylabel('脉冲PRI值',fontproperties="Songti SC", fontsize=18)

        if score is not None:
            plt.subplot(312)
            plt.plot(origin_score)
            plt.title('单向离群值打分',fontproperties="Songti SC", fontsize=18)
            # plt.xlabel('序列索引',fontproperties="Songti SC", fontsize=18)
            plt.ylabel('离群值打分',fontproperties="Songti SC", fontsize=18)

        if origin_score is not None:
            plt.subplot(313)
            plt.plot(score)
            plt.title('双向离群值打分',fontproperties="Songti SC", fontsize=18)
            plt.xlabel('序列索引',fontproperties="Songti SC", fontsize=18)
            plt.ylabel('离群值打分',fontproperties="Songti SC", fontsize=18)
        plt.show()

def display_signals_scores(signal, signal_pure, score_FSS=None, score_CUSUM=None, score_CF=None, ret = None):
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].scatter(range(len(signal)), signal, marker="+", color='b',label='原始PRI')
    ax[0].scatter(range(len(signal_pure)), signal_pure, marker="*", color='r', label='重构PRI')

    ax[0].set_ylabel("脉冲PRI值", fontproperties="Songti SC", fontsize=15)
    ax[0].set_title("PRI脉冲", fontproperties="Songti SC", fontsize=15)
    ax[0].set_xticks([])
    ax[0].tick_params(axis = 'y', which = 'major', labelsize = 15)

    ax[1].set_title("切换点检测结果", fontproperties="Songti SC", fontsize=15)
    ax[1].set_ylabel('score', fontsize=15)    
    ax[1].set_xlabel("序列索引", fontproperties="Songti SC", fontsize=15)


    if score_FSS is not None:
        ax[1].plot(score_FSS, color = 'r', label='U-FSS')
    if score_CUSUM is not None:
        ax[1].plot(score_CUSUM, color= 'g', label='U_CUSUM')
    if score_CF is not None:
        ax[1].plot(score_CF, color = 'b', label = 'ChangeFinder')
    if ret is not None:
        ax[1].plot(ret, color='b',label = 'OutlierFinder')

    ax[0].legend(loc = "best",prop={'family' : 'Songti SC', 'size'   : 15})
    ax[1].legend(loc = "best",prop={'family' : 'Times New Roman', 'size'   : 15})
    ax[1].tick_params(axis = 'y', which = 'major', labelsize = 15)
    plt.show()

def display_signal_scores(signal, pure_signal, score_FSS_dirty, score_FSS, score_CUSUM_dirty, score_CUSUM, score_CF):
    fig, ax = plt.subplots(4, 1, figsize=(17, 10))
    ax[0].plot(range(len(signal)), signal, marker="+", color='b')
    ax[0].set_ylim(0,250)
    # ax[0].set_xlabel("时间/s", fontproperties="SimHei", fontsize=12)
    ax[0].set_ylabel("脉冲PRI值", fontproperties="Songti SC", fontsize=18)
    ax[0].set_title("正弦PRI脉冲缺失脉冲3%", fontproperties="Songti SC", fontsize=18)
    ax[0].set_xticks([])
    ax[0].tick_params(axis = 'y', which = 'major', labelsize = 15)
    # axins = inset_axes(ax[0], width="50%", height="25%", loc='upper left',
    #                bbox_to_anchor=(0.25, 0.07, 0.8, 0.8),
    #                bbox_transform=ax[0].transAxes)
    # axins.scatter(range(len(signal)), signal, marker="+", color='b')
    # axins.set_xlim(200, 250)
    # axins.set_ylim(85, 97)
    # mark_inset(ax[0], axins, loc1=3, loc2=4, fc="none", ec='r', lw=1, linestyle='--')


    # plt.subplot(311)
    # plt.scatter(range(len(signal)), signal, marker='+', color='b')
    # plt.ylim(min(signal)-10, max(signal)+10)
    # plt.ylim(0, 250)

    if pure_signal is not None:
        ax[1].set_title("清洗之后的脉冲", fontproperties="Songti SC", fontsize=18)
        ax[1].scatter(range(len(pure_signal)), pure_signal, marker="+", color='r')
        ax[1].set_ylim(0, 250)
        ax[1].set_ylabel('脉冲PRI值', fontproperties="Songti SC", fontsize=18)    
        ax[1].set_xticks([])
        ax[1].tick_params(axis = 'y', which = 'major', labelsize = 15)
        # ax[1].set_xlabel("时间/s", fontproperties="SimHei", fontsize=12)

    
    # if pure_pure_signal is not None:
    #     plt.subplot(413)
    #     plt.title("清洗两次之后的脉冲", fontproperties="SimHei")
    #     plt.ylim(0, 250)
    #     plt.scatter(range(len(pure_pure_signal)), pure_pure_signal, marker='+', color='brown')
    #     # plt.xlabel('sequence')
    #     plt.ylabel('amplitude') 

    ax[2].set_title("切换点检测结果",fontproperties="Songti SC", fontsize=18)
    ax[2].set_ylabel('score', fontsize=18)    
    ax[2].set_xticks([])

    # ax[2].set_xlabel("时间/s", fontproperties="SimHei", fontsize=10)

    if score_FSS is not None:
        ax[2].plot(score_FSS_dirty, color = 'r', label='U-FSS')
    if score_CUSUM is not None:
        ax[2].plot(score_CUSUM_dirty, color= 'g', label='U-CUSUM')
    # if score_CF is not None:
    #     ax[2].plot(score_CF, color = 'b', label = 'ChangeFinder')
    ax[2].legend(loc='best',prop={'family' : 'Times New Roman', 'size'   : 15})
    ax[2].tick_params(axis = 'y', which = 'major', labelsize = 15)

    ax[3].set_title("切换点检测结果", fontproperties="Songti SC", fontsize=18)
    ax[3].set_ylabel('score', fontsize=18)    
    ax[3].set_xlabel("脉冲索引", fontproperties="Songti SC", fontsize=18)


    if score_FSS is not None:
        ax[3].plot(score_FSS, color = 'r', label='OF+CDA_1')
    if score_CUSUM is not None:
        ax[3].plot(score_CUSUM, color= 'g', label='OF+CDA_2')
    if score_CF is not None:
        ax[3].plot(score_CF, color = 'b', label = 'ChangeFinder')

    ax[3].legend(loc = "best",prop={'family' : 'Times New Roman', 'size'   : 15})
    ax[3].tick_params(axis = 'y', which = 'major', labelsize = 15)
    plt.show()

def display_signals_scores_clean_twice(signal, pure_signal, pure_pure_signal, score_FSS, score_CUSUM, score_CF):
    fig, ax = plt.subplots(4, 1, figsize=(17, 10))
    ax[0].scatter(range(len(signal)), signal, marker="+", color='b')
    ax[0].set_ylim(0,250)
    # ax[0].set_xlabel("时间/s", fontproperties="SimHei", fontsize=12)
    ax[0].set_ylabel("脉冲PRI值", fontproperties="Songti SC", fontsize=18)
    ax[0].set_title("非理想高斯抖动PRI脉冲15%", fontproperties="Songti SC", fontsize=18)
    ax[0].set_xticks([])
    ax[0].tick_params(axis = 'y', which = 'major', labelsize = 15)


    if pure_signal is not None:
        ax[1].set_title("清洗之后的脉冲", fontproperties="Songti SC", fontsize=18)
        ax[1].scatter(range(len(pure_signal)), pure_signal, marker="+", color='r')
        ax[1].set_ylim(0, 250)
        ax[1].set_ylabel('脉冲PRI值', fontproperties="Songti SC", fontsize=18)    
        ax[1].set_xticks([])
        ax[1].tick_params(axis = 'y', which = 'major', labelsize = 15)

    
    if pure_pure_signal is not None:
        ax[2].set_title("清洗两次之后的脉冲", fontproperties="Songti SC", fontsize=18)  
        ax[2].scatter(range(len(pure_pure_signal)), pure_pure_signal, marker="+", color='r')

        ax[2].text(-6, 200, r"$(\hat{\mu}=49,\hat{\sigma^2}=6.2)$", font_title)
        ax[2].annotate('', xy=(47,180), xytext=(47,100), arrowprops=dict(facecolor='black', shrink=0.1, width=2))

        ax[2].text(133, 200, r"$(\hat{\mu}=88,\hat{\sigma^2}=7.5)$",font_title)
        ax[2].annotate('', xy=(188,180), xytext=(188,100), arrowprops=dict(facecolor='black', shrink=0.1, width=2))

        ax[2].text(265, 200, r"$(\hat{\mu}=73,\hat{\sigma^2}=7.1)$",font_title)
        ax[2].annotate('', xy=(320,180), xytext=(320,100), arrowprops=dict(facecolor='black', shrink=0.1, width=2))

        ax[2].text(390, 200, r"$(\hat{\mu}=81,\hat{\sigma^2}=7.2)$",font_title)
        ax[2].annotate('', xy=(442,180), xytext=(442,100), arrowprops=dict(facecolor='black', shrink=0.1, width=2))

        ax[2].text(520, 200, r"$(\hat{\mu}=62,\hat{\sigma^2}=6.5)$",font_title)
        ax[2].annotate('', xy=(570,180), xytext=(570,100), arrowprops=dict(facecolor='black', shrink=0.1, width=2))

        # ax[2].text(-6, 200, r"$(\hat{a}=30,\hat{b}=70)$", font_title)
        # ax[2].annotate('', xy=(47,180), xytext=(47,100), arrowprops=dict(facecolor='black', shrink=0.1, width=2))

        # ax[2].text(133, 200, r"$(\hat{a}=85,\hat{b}=102)$",font_title)
        # ax[2].annotate('', xy=(188,180), xytext=(188,100), arrowprops=dict(facecolor='black', shrink=0.1, width=2))

        # ax[2].text(275, 200, r"$(\hat{a}=68,\hat{b}=78)$",font_title)
        # ax[2].annotate('', xy=(320,180), xytext=(320,100), arrowprops=dict(facecolor='black', shrink=0.1, width=2))

        # ax[2].text(390, 200, r"$(\hat{a}=76,\hat{b}=93)$",font_title)
        # ax[2].annotate('', xy=(442,180), xytext=(442,100), arrowprops=dict(facecolor='black', shrink=0.1, width=2))

        # ax[2].text(520, 200, r"$(\hat{a}=58,\hat{b}=81)$",font_title)
        # ax[2].annotate('', xy=(570,180), xytext=(570,100), arrowprops=dict(facecolor='black', shrink=0.1, width=2))

        ax[2].set_ylim(0, 250)
        ax[2].set_ylabel('脉冲PRI值', fontproperties="Songti SC", fontsize=18)    
        ax[2].set_xticks([])
        ax[2].tick_params(axis = 'y', which = 'major', labelsize = 15)


    ax[3].set_title("切换点检测结果", fontproperties="Songti SC", fontsize=18)
    ax[3].set_ylabel('score', fontsize=18)    
    ax[3].set_xlabel("脉冲索引", fontproperties="Songti SC", fontsize=18)


    if score_FSS is not None:
        ax[3].plot(score_FSS, color = 'r', label='OF+CDA_1')
    if score_CUSUM is not None:
        ax[3].plot(score_CUSUM, color= 'g', label='OF+CDA_2')
    if score_CF is not None:
        ax[3].plot(score_CF, color = 'b', label = 'ChangeFinder')

    ax[3].legend(loc = "best",prop={'family' : 'Times New Roman', 'size'   : 15})
    ax[3].tick_params(axis = 'y', which = 'major', labelsize = 15)
    plt.show()

def display_signal_clean(signal, signal_pure):
    plt.subplot(211)
    plt.scatter(range(len(signal)), signal, marker='+', color='b')
    plt.ylim(0, 200)
    plt.title('signal')
    plt.xlabel('sequence')
    plt.ylabel('amplitude')

    plt.subplot(212)
    plt.scatter(range(len(signal_pure)), signal_pure, marker='+', color='r')
    plt.ylim(0, 200)
    plt.title('cleaned_signal')
    plt.xlabel('sequence')
    plt.ylabel('amplitude')
    plt.show()

# def display_signal_scores_bic(signal, signal_pure = None, score_FSS = None, score_CUSUM = None, score_CF = None, bic):
#     fig, ax = plt.subplots(4, 1, figsize=(17, 10))
#     ax[0].scatter(range(len(signal)), signal, marker="+", color='b')
#     ax[0].set_ylim(0,250)
#     ax[0].set_ylabel("脉冲PRI值", fontproperties="Songti SC", fontsize=18)
#     ax[0].set_title("正弦PRI脉冲缺失脉冲3%", fontproperties="Songti SC", fontsize=18)
#     ax[0].set_xticks([])
#     ax[0].tick_params(axis = 'y', which = 'major', labelsize = 15)

#     if pure_signal is not None:
#         ax[1].set_title("清洗之后的脉冲", fontproperties="Songti SC", fontsize=18)
#         ax[1].scatter(range(len(pure_signal)), pure_signal, marker="+", color='r')
#         ax[1].set_ylim(0, 250)
#         ax[1].set_ylabel('脉冲PRI值', fontproperties="Songti SC", fontsize=18)    
#         ax[1].set_xticks([])
#         ax[1].tick_params(axis = 'y', which = 'major', labelsize = 15)

#     ax[2].set_title("切换点检测结果", fontproperties="Songti SC", fontsize=18)
#     ax[2].set_ylabel('score', fontsize=18)    
#     ax[2].set_xlabel("脉冲索引", fontproperties="Songti SC", fontsize=18)


#     if score_FSS is not None:
#         ax[2].plot(score_FSS, color = 'r', label='OF+CDA_1')
#     if score_CUSUM is not None:
#         ax[2].plot(score_CUSUM, color= 'g', label='OF+CDA_2')
#     if score_CF is not None:
#         ax[2].plot(score_CF, color = 'b', label = 'ChangeFinder')

#     ax[2].legend(loc = "best",prop={'family' : 'Times New Roman', 'size'   : 15})
#     ax[2].tick_params(axis = 'y', which = 'major', labelsize = 15)

#     ax[3].set_title("切换点检测结果",fontproperties="Songti SC", fontsize=18)
#     ax[3].set_ylabel('score', fontsize=18)    
#     ax[3].set_xticks([])

#     # ax[2].set_xlabel("时间/s", fontproperties="SimHei", fontsize=10)

#     if score_FSS is not None:
#         ax[3].plot(score_FSS_dirty, color = 'r', label='U-FSS')
#     if score_CUSUM is not None:
#         ax[3].plot(score_CUSUM_dirty, color= 'g', label='U-CUSUM')
#     # if score_CF is not None:
#     #     ax[2].plot(score_CF, color = 'b', label = 'ChangeFinder')
#     ax[3].legend(loc='best',prop={'family' : 'Times New Roman', 'size'   : 15})
#     ax[3].tick_params(axis = 'y', which = 'major', labelsize = 15)
#     plt.show()
  

def PR_cruve(bkps, ret):
    #真实标签
    y_true = [0] * len(ret)
    for point in bkps[:-2]:
        y_true[point] = 1
    y_true = np.array(y_true)
    #建立概率标签
    y_score = []
    for index in range(len(ret)):
        y_score.append(ret[index]/max(ret))
    y_score = np.array(y_score)

    for index in range(len(ret)):
        print(y_true[index], y_score[index])


    precision, recall, threshold = precision_recall_curve(y_true, y_score)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision/Recall Curve")
    plt.plot(recall, precision)
    plt.show()

def benefit_alarm_curve():
    x1 = [0, 0.1, 0.2,  0.4,  0.42,  0.5,   0.57, 0.81]#点的横坐标
    x2 = [0, 0.1, 0.25, 0.4,  0.57,  0.66,  0.7,  0.8]
    x3 = [0, 0.1, 0.6,  0.7,  0.81,  0.85,  0.9,  0.93]
    # x4 = [0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.7, 0.95]
    k1 = [0.22, 0.4,  0.4,  0.4,  0.45,  0.45,  0.47,  0.6]#线1的纵坐标
    k2 = [0.22, 0.43, 0.5,  0.6,  0.63,  0.7,   0.73,  0.78]#线2的纵坐标
    k3 = [0.16,  0.25, 0.28, 0.3,  0.31,  0.25,  0.3,  0.4]
    # k4 = [0.76, 0.92, 0.925, 0.93, 0.94, 0.94, 0.95, 0.99]
    plt.plot(x1,k1,'s-',color = 'r',label="OF+CDA_1")#s-:方形
    plt.plot(x2,k2,'o-',color = 'g',label="OF+CDA_2")#o-:圆形
    plt.plot(x3,k3,'bd-',color = 'b',label="ChangeFinder")
    # plt.plot(x4,k4,'go-', color= 'brown', label='ChangeFinder_ARIMA')
    plt.xlabel("虚警率",fontproperties="Songti SC", fontsize=18)#横坐标名字
    plt.ylabel("平均效益", fontproperties="Songti SC", fontsize=18)#纵坐标名字
    plt.title("平均效益虚警率对抗图",  fontproperties="Songti SC", fontsize=18)

    plt.legend(loc = "best",prop={'family' : 'Times New Roman', 'size'   : 15})#图例
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()

def benefit_alarm_curve_nonco():
    x1 = [0, 0, 0.2, 0.42, 0.55, 0.69, 0.86]#点的横坐标
    x2 = [0, 0.1, 0.2, 0.55, 0.6, 0.73, 0.86]
    x3 = [0, 0.1, 0.1, 0.3, 0.3, 0.6, 0.9]
    # x4 = [0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.7, 0.95]
    k1 = [1,0.625,0.6,0.625,0.65,0.75,0.625]#线1的纵坐标
    k2 = [0.6, 0.75, 0.95, 0.95, 0.95, 0.975, 0.975]#线2的纵坐标
    k3 = [0.75,0.76,0.88,0.88,0.9,0.92,0.93]
    # k4 = [0.76, 0.92, 0.925, 0.93, 0.94, 0.94, 0.95, 0.99]
    plt.plot(x1,k1,'s-',color = 'r',label="U-FSS")#s-:方形
    plt.plot(x2,k2,'o-',color = 'g',label="U-CUSUM")#o-:圆形
    plt.plot(x3,k3,'bd-',color = 'b',label="ChangeFinder")
    # plt.plot(x4,k4,'go-', color= 'brown', label='ChangeFinder_ARIMA')
    plt.xlabel("false alarm rate", fontsize=10)#横坐标名字
    plt.ylabel("average benefit", fontsize=10)#纵坐标名字
    plt.legend(loc = "best")#图例
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()

def algorithm_delay():
    x = ('FSS', 'CUSUM', 'Change\nFinder', 'Change\nFinder\nARIMA', 'non \n cooperative \n framework')
    time = [37.1, 27, 96, 200, 70.4]
    plt.bar(x, time)
    plt.ylabel('average detection time/ms')
    plt.show()

def paint_gaussian_distribution():
    u0 = 0
    u1 = 1
    sig = math.sqrt(1)
    x0 = np.linspace(u0 - 3*sig, u0 + 3*sig, 50)
    y0 = np.exp(-(x0 - u0) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig)
    x1  = np.linspace(u1 - 3*sig, u1 + 3*sig, 50)
    y1 = np.exp(-(x1 - u1) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig)

    plt.figure(num=3,figsize=(9,6))
    plt.plot(x1, y1, 'black', linewidth=2)
    plt.plot(x0, y0, 'black', linewidth=2)
    plt.xlabel('x[0]')


    new_ticks = np.linspace(-4,4,9)
    plt.xticks(new_ticks)
    plt.yticks([])

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')


    ax.spines['bottom'].set_position(('data',0))
    ax.spines['left'].set_position(('data',0))

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.axvline(x=0.5, c = 'black', ls='--',lw=1)

    
    # plt.arrow(1.2,1.3,0.5,0.3,width=0.1,fc="b")


    plt.show()

def compare_chart():
    def optimal_FSS(x, K):
        y = 2*math.log(x/K) / 0.5
        return y
        
    T = np.linspace(100, 10**10, num = 1000000, dtype=int)

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(10**2, 10**11)
    for K in [2, 5, 10, 20, 50]:
        tau = []
        color = {2:'blue', 5:'red', 10:'black', 20:'green', 50:'grey'}
        for x in T:
            y = optimal_FSS(x, K)
            tau.append(y)
        plt.plot(tau, T, c='black')

    # plt.xlim(10**0, 10**2)
    plt.show()

if __name__ == '__main__':
    # try:
    #     from dataset import datasets
    # except ImportError:
    #     print('import error')
    # data = datasets(1000, 10)
    # signal, bkps = data.jumping_mean_random(1000, 10)
    # display_signal_score(signal)

    benefit_alarm_curve()

    # paint_gaussian_distribution()

    # compare_chart()

    # algorithm_delay()
    