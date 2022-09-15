'''
author: Jiadi
Date: 2022.5.7
TODO: add parser
'''
import random
from math import ceil
import numpy as np
import pylab as P
from scipy.stats import entropy
# from torch import threshold

def _blob(x,y,area,colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    P.fill(xcorners, ycorners, colour, edgecolor=colour)

def hinton(W, maxWeight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix. 
    Temporarily disables matplotlib interactive mode if it is on, 
    otherwise this takes forever.
    """
    reenable = False
    if P.isinteractive():
        P.ioff()
    P.clf()
    height, width = W.shape
    if not maxWeight:
        maxWeight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))

    P.fill(np.array([0,width,width,0]),np.array([0,0,height,height]),'gray')
    P.axis('off')
    P.axis('equal')
    for x in range(width):
        for y in range(height):
            _x = x+1
            _y = y+1
            w = W[y,x]
            if w > 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,w/maxWeight),'white')
            elif w < 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,-w/maxWeight),'black')
    if reenable:
        P.ion()
    P.show()

def gen_alpha(type = 'jitter', K = 3):
    '''
    为HMM生成状态转移矩阵和HMM集群中心
    type:调制基形,以PRI为例,可选的调制基形:
    'jitter','stagger','slide','dwell'
    center:不同调制基形对应不同集群中心,如果不指定集群中心,默认中心个数=3

    默认均为一维变量
    '''

    if type == 'jitter':
        assert K==1, "K must be 1"
        alpha = np.eye(K)
    if type in ['stagger', 'agile', 'dwell']:
        alpha = np.eye(K)
        while(1 in np.diag(alpha)):
            np.random.shuffle(alpha)
    if type == 'slide':
        alpha = np.eye(K)
        alpha = np.vstack((alpha[1:],alpha[0]))
    # if type == 'dwell':
    #     tr = np.eye(K)
    #     while(1 in np.diag(tr)):
    #         np.random.shuffle(tr)
    #     alpha = 0.05*tr + np.eye(K)
    return alpha

def Non_ideal_HMM(N, K, XDim=1, mu_sd_factor=0, L=10, spur_ratio=0, miss_ratio=0,type='stagger', mu = None):
    '''N: number of observations
    K: number of discrete components
    XDim: dimensionality of observed data (default = 2)
    mu:HMM集群的中心值，如果没有制定就是[100, 110, 120,...]

    计算虚假脉冲和缺失脉冲个数'''

    spur_num = np.floor(spur_ratio * N)
    miss_num = np.floor(miss_ratio * N)
    N_unideal = int(N + spur_num + miss_num)
    X = np.zeros((N_unideal,XDim))
    Y = np.zeros((N_unideal,L))
    z = np.zeros(K)#记录上一个状态
    Z = np.zeros((N_unideal, 1))
    alpha = gen_alpha(type=type, K=K)

    beta = 0.1 #intial dist params.  
    pik = np.random.dirichlet(beta*np.ones(K)) #distribution over initial state p(l_0)
    A = alpha / np.reshape(alpha.sum(axis=1),(K,1))#把alpha归一化
    # mu = np.array([np.random.multivariate_normal(np.zeros(XDim),20*np.eye(XDim)) for _ in range(K)]) #每个类别的均值
    if mu == None:
        mu = np.array([[100+10*i] for i in range(K)])
        mu_init = mu.copy()
    else:
        mu = np.array([mu])
        print(mu)
        mu = mu.T
        mu_init = mu.copy()
    C = np.array([mu_sd_factor*np.eye(XDim) for _ in range(K)])#每个类别的方差
    pr_y = np.array([np.random.dirichlet(0.1*np.ones(L)) for _ in range(K)])#每个类别的可能性的先验概率
    #draw observations:
    n = 0
    bound = X.shape[0]
    while(n<=bound):
        #draw latent component:
        if n==0:
            z = np.random.multinomial(1, pik)
        else:
            prev_z = z.argmax()
            z = np.random.multinomial(1,A[prev_z,:])
        z_n = z.argmax()#记录下一个状态
        if np.random.choice(2, 1, p=[1 - miss_ratio, miss_ratio]) and type is not 'dwell':#缺失脉冲
            X = np.delete(X, n, axis=0)
            Y = np.delete(Y, n, axis=0)
            Z = np.delete(Z, n, axis=0)
            continue
        if np.random.choice(2, 1, p=[1 - spur_ratio, spur_ratio]) and type is not 'dwell':#虚假脉冲
            spur_value = np.random.uniform(low=min(mu),high=max(mu),size=1)
            X[n,:] = np.random.multivariate_normal(spur_value, C[z_n,:,:])
            # Z[n,:] = abs(mu-spur_value).argmin()
            Z[n,:] = K+1
            n+=1
        if type=='dwell':
            for dns in range(5):
                if np.random.choice(2, 1, p=[1 - miss_ratio, miss_ratio]):#缺失脉冲
                    X = np.delete(X, n, axis=0)
                    Y = np.delete(Y, n, axis=0)
                    Z = np.delete(Z, n, axis=0)
                    continue
                if np.random.choice(2, 1, p=[1 - spur_ratio, spur_ratio]):#虚假脉冲
                    spur_value = np.random.uniform(low=min(mu),high=max(mu),size=1)
                    X[n,:] = np.random.multivariate_normal(spur_value, C[z_n,:,:])
                    # Z[n,:] = abs(mu-spur_value).argmin()
                    Z[n,:] = K+1
                    n+=1
                X[n,:] = np.random.multivariate_normal(mu[z_n,:], C[z_n,:,:])
                Y[n,:] = np.random.multinomial(1,pr_y[z_n,:])
                Z[n,:] = z_n
                bound = X.shape[0]-1
                n+=1
                if n>bound:
                    break
        else:
            X[n,:] = np.random.multivariate_normal(mu[z_n,:], C[z_n,:,:])
            if type == 'agile': 
                Z[n,:] = np.where(mu_init == mu[z_n,:])[0]
            if (n+1) % K == 0 and type=='agile':
                np.random.shuffle(mu)

        if type !='dwell':
            Y[n,:] = np.random.multinomial(1,pr_y[z_n,:])
            if type != 'agile':
                Z[n,:] = z_n
            n+=1
            bound = X.shape[0]-1
    #返回每个类别的均值
    return X,Y,Z,mu

def workmode_cat(nums = None, Ks = None, mus = None, XDim=1, mu_sd_factor = 0, spur_ratio = 0, miss_ratio = 0, types = None):
    '''
    将生成的数据拼接
    nums Ks types均为列表 长度相同
    '''
    seq_num = len(nums)
    assert len(Ks) == seq_num, "Do not have enough Ks"
    assert len(types) == seq_num, "Do not have enough types"
    assert len(mus) == seq_num, "Do not have enough mus"

    X = None
    Z = None
    mus_grd = None
    paras = zip(nums, Ks, mus, types)
    for para in paras:
        session, _, labels, mu_grd = Non_ideal_HMM(N=para[0], K=para[1],mu = para[2], XDim=XDim, mu_sd_factor=mu_sd_factor, spur_ratio=spur_ratio, miss_ratio=miss_ratio,type=para[3])
        if X is None and mus_grd is None:
            X = session
            Z = labels
            mus_grd = mu_grd
        else:
            X = np.vstack((X, session))
            Z = np.vstack((Z, labels))
            mus_grd = np.vstack((mus_grd, mu_grd))
    return X, Z, mus_grd

def center_compare(center1, center2, threshold):
    '''
    比较两个集群中心
    '''
    diff = np.array([])
    for num in center1:
        diff = np.append(diff,min(abs(num-center2)))
    change = diff.sum()/diff.shape[0] > threshold
    cusum =  float(diff.sum())/diff.shape[0]
    return change,threshold #1是手动设置阈值

def trans_compare(A_1, A_2, mu_1, mu_2,threshold=1):
    def trans(A, order):
        N = len(order)
        # print(order)
        new_matrix = np.zeros((N,N))
        for i, row in enumerate(order):
            for j, col in enumerate(order):
                try:
                    new_matrix[row][col] = A[i][j]
                except IndexError:
                    if A.shape[0] < N:#A_1<A_2,填上0
                        new_matrix[row][col] = 0.1
                    # else:

        return new_matrix

    def trans_order(mu_1,mu_2):
        order = []
        N = len(mu_1)
        minus_dis = []
        for mu in mu_1:
            minus = mu_2 - mu
            order.append(np.argmin(abs(minus)))
            minus_dis.append(np.min(abs(minus)))
        return order

    order = trans_order(mu_1,mu_2)
    A2_new = trans(A_2,order)
    e = 0
    for index in range(len(order)):
        e += entropy(A_1[index], A2_new[index])
    change = e > threshold
    cusum = e
    return change, cusum

if __name__=="__main__":
    
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    import pandas as pd
    np.set_printoptions(threshold=np.inf)
    plt.rc('font',family='Times New Roman')

    #数据生成
    N, K_grnd, K = 100, 5, 10
    X1, _, Z, mu_grd = Non_ideal_HMM(N, K_grnd, mu_sd_factor= 0, spur_ratio=0., miss_ratio=0,type='dwell')
    
    # Ns = [100,150,180]
    # Ks = [4,5,6]
    # types = ['stagger','slide','stagger']
    # X, mu_grd = workmode_cat(nums=Ns, Ks=Ks, mu_sd_factor=0, spur_ratio=0, miss_ratio=0, types=types)
    pulse = []
    labels = []
    for p, z in zip(X1,Z):
        labels.append(int(z))
        pulse.append(int(p))
    
    data = [range(len(pulse)),pulse,labels]
    data = zip(*data)
    df = pd.DataFrame(data,columns=['index','X','Z'])

    #颜色赋值
    col = []
    colors = ['#FF0000', '#FFA500', 'purple', 'blue', '#228B22']
    for i in range(0, len(labels)):
        col.append(colors[int(labels[i])])



    # plt.figure(dpi=150,figsize=(16,5))
    # plt.xlabel("PRI index")
    # plt.ylabel("PRI value")
    # plt.title("agile PRI")
    plt.axis('off')
    # plt.plot(range(len(X1)), X1, markersize=6, marker='*',c=col)  
    df.plot.scatter('index', 'X', c=col, figsize=(16,5),s=6)
    df.plot('index', 'X', c='black')

    # plt.plot(Z+120)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)

    plt.savefig("genworkmode/dwell_color.png", transparent=True, dpi=300, pad_inches = 0,format='png')
    plt.show()
