import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import statsmodels.api as sm
from AR_model import SDAR
from hmmlearn import hmm
import time
import seaborn as sns
from functools import reduce
sns.set_theme(style="darkgrid")


class ChangeFinderAbstract(object):
    '''
    one下一个打分
    ts为储存长度为T打分的列表
    size为平滑长度T

    add_one方法：将下一个打分添加到ts打分列表中
    smoothing方法：返回y_t，为长度size的平均值，也就是Taverage score
    '''
    def add_one(self, one, ts, size):
        ts.append(one)
        if len(ts) == size + 1:
            ts.pop(0)

    def smoothing(self, ts):
        return sum(ts) / float(len(ts))

class ChangeFinder(ChangeFinderAbstract):
    '''
    为ChangeFinderAbstract的子类

    r为
    order为AR模型的阶数
    smooth为 Taverage score 的T，平滑长度
    ts为score的列表

    first_scores为第一次打分
    second_scores为第二次打分

    smoothed_scores为Taverage score

    sdar_first为outlier detection
        update方法可以返回当前概率密度函数
    sdar_second为changepoint detection
        update方法可以返回当前概率密度函数

    ChangeFinder的update方法
    '''
    def __init__(self, r=0.01, order=1, smooth=7, outlier = False):
        assert order > 0, "order must be 1 more"
        assert smooth > 2, "term must be 3 or more"

        self.smooth = smooth
        self.smooth2 = int(round(self.smooth / 2.0))

        self.order = order
        self.r = r
        self.ts = []
        self.first_scores = []
        self.second_scores = []
        self.smoothed_scores = []
        self.sdar_first = SDAR(r, self.order)
        self.sdar_second = SDAR(r, self.order)
        self.outlier = outlier

    def update(self, x):
        #打分
        score = 0

        #序列的下一个值
        predict = x
        predict2 = 0

        #向ts中加入序列中的下一个数，直到ts中的数量和order相等
        if len(self.ts) == self.order:
            score, predict = self.sdar_first.update(x, self.ts)
            #更新打分值和预测值之后，将打分放入firstscores，一次放入smooth个
            self.add_one(score, self.first_scores, self.smooth)
        #order = 1 一次放一个
        self.add_one(x, self.ts, self.order)

        #second_target是Taverage score
        second_target = None

        #每加入一个数计算一个分数，如果分数等于T，则取平均值，即为second_target
        if len(self.first_scores) == self.smooth:
            second_target = self.smoothing(self.first_scores)

        if self.outlier==True:
            if score>8:
                self.ts[-1]=predict
            return score, predict

        #如果secondtarget被计算了，整好采够了smoothedscores
        if second_target and len(self.smoothed_scores) == self.order:
            #第二次学习，打分，预测
            score, predict2 = self.sdar_second.update(second_target, self.smoothed_scores)
            #学习的结果放入secondscores，放Smooth2个
            self.add_one(score, self.second_scores, self.smooth2)
        #如果没有满，则将secondtarget加入smoothedscores，一次加一个
        if second_target:
            self.add_one(second_target, self.smoothed_scores, self.order)
        #当secondscores里面的个数等于smooth2，平滑
        if len(self.second_scores) == self.smooth2:
            return self.smoothing(self.second_scores), predict
        else:
            return 0.0, predict

class ChangeFinder_ARIMA(ChangeFinderAbstract):
    
    def __init__(self, term=30, smooth=7, order=(1,0,0), outlier=False):
        assert smooth > 2, "term must be 3 or more."
        assert term > smooth, "term must be more than smooth"

        #term用来存放数据段
        self.term = term
        #取平均值
        self.smooth = smooth
        self.smooth2 = int(round(self.smooth/2.0))
        #(p,d,q),pq为movingaverage的起点和终点，d为阶数
        self.order = order
        #ts来存放数据，每次addone
        self.ts = []
        self.first_scores = []
        self.smoothed_scores = []
        self.second_scores = []
        self.outlier = False
    
    def calculate_outlier(self, ts, target):
        def outlier_score(residuals, x):
            #取残差的平均
            m = residuals.mean()
            #计算残差的标准差
            s = np.std(residuals, ddof=1)
            #返回对数概率密度，x为概率密度函数的横坐标
            return -sp.stats.norm.logpdf(x, m, s)
        ts = np.array(ts)
        arima_model = sm.tsa.ARIMA(ts, self.order)
        result = arima_model.fit(disp=0)
        #计算预测值
        pred = result.forecast(1)[0][0]
        #返回对数概率密度和预测值
        return outlier_score(result.resid, x=pred-target), pred

    def update(self, x):
        score = 0
        predict = x
        predict2 = 0

        if len(self.ts) == self.term:
            #当ts中存储的时间序列数据长度为term时
            try:
                #x为下一个输入数据
                #这句话的意思时，已知ts中的先验信息，x出现的对数概率密度，也是score
                score, predict = self.calculate_outlier(self.ts, x)
                #在first_scores里加入新的score，如果超过平滑极限，则删除第一个
                self.add_one(score, self.first_scores, self.smooth)
            except Exception:
                #如果要是到了序列末尾，在ts中加入
                self.add_one(x, self.ts, self.term)
                return 0, predict
        #向ts列表中加入x
        self.add_one(x, self.ts,self.term)

        second_target = None

        if len(self.first_scores) == self.smooth:
            #如果第一次学习得到的score达到了平滑最少的数量 7个
            second_target = self.smoothing(self.first_scores)
            
        if self.outlier:
            return score, predict

        if second_target and len(self.smoothed_scores) == self.term:
            #当第一次学习的结果达到term的时候，用这些结果进行第二次学习
            try:
                score, predict2  = self.calculate_outlier(self.smoothed_scores, second_target)
                self.add_one(score, self.second_scores, self.smooth2)
            except Exception:
                self.add_one(second_target, self.smoothed_scores, self.term)
                return 0, predict
        if second_target:
            self.add_one(second_target, self.smoothed_scores, self.term)
        if len(self.second_scores) == self.smooth2:
            #进行第二次平滑
            return self.smoothing(self.second_scores), predict
        else:
            return 0.0, predict

class ChangeFinder_HMM(ChangeFinderAbstract):
    def __init__(self, space = 20, smooth=7, outlier=False):
        assert space > 14, "order must be 14 more"
        assert smooth > 2, "term must be 3 or more"

        self.smooth = smooth
        self.smooth2 = int(round(self.smooth / 2.0))

        self.space = space
        self.r = r
        self.ts = []
        self.first_scores = []
        self.second_scores = []
        self.smoothed_scores = []
        self.model = hmm.GaussianHMM(n_components=3, covariance_type="diag")
        self.model2 = hmm.GaussianHMM(n_components=3, covariance_type="diag")
        self.outlier = outlier

    def update(self, x):
        
        score = 0

        predict = x
        predict2 = 0

        if len(self.ts) == self.space:
            self.model.fit(ts)
            score= self.model.score(self.ts)
            predict = self.model.sample(1)
            #更新打分值和预测值之后，将打分放入firstscores，一次放入smooth个
            self.add_one(score, self.first_scores, self.space)
        #order = 1 一次放一个
        self.add_one(x, self.ts, self.space)

        #second_target是Taverage score
        second_target = None

        #每加入一个数计算一个分数，如果分数个数等于T，则取平均值，即为second_target
        if len(self.first_scores) == self.smooth:
            second_target = self.smoothing(self.first_scores)

        if self.outlier==True:
            return score, predict
        
        #如果secondtarget被计算了，整好采够了smoothedscores
        if second_target and len(self.smoothed_scores) == self.space:
            self.model2.fit(self.smoothed_scores)
            #第二次学习，打分，预测
            score = self.model2.score(second_target)
            predict = self.model2.sample(1)
            #学习的结果放入secondscores，放Smooth2个
            self.add_one(score, self.second_scores, self.smooth2)
        #如果没有满，则将secondtarget加入smoothedscores，一次加一个
        if second_target:
            self.add_one(second_target, self.smoothed_scores, self.order)
        #当secondscores里面的个数等于smooth2，平滑
        if len(self.second_scores) == self.smooth2:
            return self.smoothing(self.second_scores), predict
        else:
            return 0.0, predict

class CUsum(object):
    def __init__(self, bkps=[], mean=[], var=[], para_known=False, changepoint_th=20):
        #用来逐一加入数据
        self.ts = [] 
        #用来逐一加入数据，每次检测到切换点之后更新
        self.ts_temp = []
        #存放累计和
        self.s_kt = []
        #存放反向累计和
        self.s_kt_reverse = []
        #输出切换点
        self.bkp_detect = []
        #存放真实切换点
        self.bkps = bkps
        #存放真实均值
        self.mean = mean
        #存放真实方差
        self.var = var
        #在中间列表中用来计数
        self.index = 0
        #存放真实的均值
        self.average = []
        self.changepoint_th = changepoint_th
        self.para_known = para_known
        j=0
        for i in range(len(mean)):
            while j < bkps[i]:
                self.average.append(mean[i])
                j+=1
        
    def add_one(self, ts, one):
        ts.append(one)

    def calculate_threshold(self, theta_1, theta_2):
        length = len(self.ts_temp)
        threshold = -2 * (theta_2 - theta_1) * sum(self.ts_temp) + length * (np.square(theta_2) - np.square(theta_1)) 
        return threshold*2

    def calculate_log_likelihood_ratio(self, ave, x):
        length = len(self.ts_temp)
        LLR = -2 * (x - ave) * sum(self.ts_temp) + length * (np.square(x) - np.square(ave))
        return LLR

    def update(self, x):
        if self.para_known == True:
            self.add_one(self.ts, x)
            self.add_one(self.ts_temp, x)

            ave = self.average[len(self.ts)-2]
            s = time.time()
            for mask in self.average[(len(self.ts)-1):]:
                if mask != ave:
                    ave_next = mask
                    break
                else:
                    ave_next = self.mean[-1] * 1000
            # print(time.time()-s)
            # print("!!!!!!!!!!!!!!!!!!!!")
            threshold = self.calculate_threshold(ave, ave_next)
            LLR = self.calculate_log_likelihood_ratio(ave, x)
            self.s_kt.append(LLR)
            # print(ave, x, ave_next, LLR, threshold)
            if abs(sum(self.s_kt)) >= abs(threshold):
                self.bkp_detect.append(len(self.s_kt))
                # print("changeeeeeeeeeeeeeee")
                # print(len(self.s_kt))
                self.ts_temp = []
                self.s_kt = []
                return 1
            
            return 0
            
        if self.para_known == False:
            self.add_one(self.ts, x)
            self.add_one(self.ts_temp, x)

            ave = np.mean(self.ts_temp)

            threshold = self.calculate_threshold(ave, ave+self.changepoint_th)
            LLR = self.calculate_log_likelihood_ratio(ave, x)
            self.s_kt.append(LLR)

            if abs(sum(self.s_kt)) > abs(threshold):
                self.bkp_detect.append(len(self.s_kt))
                # print("changeeeeeeeeeeeeeee")
                print(len(self.s_kt))
                self.ts_temp = []
                self.s_kt = []
                return 10
            return 0

class CUSUM_BNP_HMM(object):
    def __init__(self, bkps=[], mean=[], var=[], para_known=False, threshold=20):
        #用来逐一加入数据
        self.ts = [] 
        #用来逐一加入数据，每次检测到切换点之后更新
        self.ts_temp = []
        #存放累计和
        self.s_kt = []
        #输出切换点
        self.bkp_detect = []
        #存放真实切换点
        self.bkps = bkps
        #存放真实均值
        self.mean = mean
        #存放真实方差
        self.var = var
        #在中间列表中用来计数
        self.index = 0
        #generalized hypergeometric function自由度
        self.r = 10
        #存放真实的均值
        self.average = []
        self.threshold = threshold
        self.para_known = para_known

        j=0
        for i in range(len(mean)):
            while j < bkps[i]:
                self.average.append(mean[i])
                j+=1
        
    def add_one(self, ts, one):
        ts.append(one)
    
    def lnG(self, d, z, order=5):
        """
        generalized hypergeometric function
        """
        item = []
        for n in range(order):
            if n == 0:
                item.append(1)
            else:
                numerator = np.power(z, n)
                denominator = reduce(lambda x,y:x*y, [d+i for i in range(n)]) * np.math.factorial(n)
                item.append(numerator / denominator)
        return np.log(sum(item))
    
    def b_square(self, theta_0, theta):
        '''
        theta_0是切换前的参数,使用DPMM拟合
        theta是切换后的参数,通过BNP-HMM估计
        输入时保证两个参数维度相同
        '''
        diff = np.array([])
        for num in theta:
            diff = np.append(diff,min(abs(num-theta_0)))
        return np.dot(np.transpose(diff),diff)
    
    def chi_square(self, theta_0, ave):
        '''
        计算chi_k^t
        '''
        diff = np.array([])
        for num in ave:
            minus =  num-theta_0
            trg = np.argmin(abs(minus))
            diff = np.append(diff, minus[trg])
        return np.dot(np.transpose(diff),diff)
    
    def calculateS_kt(self, theta_0, ave, pulse):
        S_kt = 0
        length = len(self.ts_temp)
        # b2 = self.b_square(theta_0, pulse)
        b2 = 125
        chi2 = self.chi_square(theta_0, ave)
        S_kt += -length * b2
        S_kt += self.lnG(self.r/2, b2 * np.power(length, 2) * chi2/4)
        return S_kt

    def update(self, x):
        self.add_one(self.ts, x)
        self.add_one(self.ts_temp, x)#用来存放Skt
        buffer = []
        s_kt_buffer = []
        for pulse in self.ts_temp:
            self.add_one(buffer, pulse)
            ave = np.mean(buffer, axis=0)
            theta_0 = self.ts_temp[0]
            print(theta_0, ave)
            chi2 = self.chi_square(theta_0, ave)
            print(chi2)
            s_kt_buffer.append(chi2*len(self.ts_temp)/2)
        self.s_kt.append(sum(s_kt_buffer))
        # print(self.s_kt)
            # S_kt = self.calculateS_kt(theta_0, ave, pulse)
            # self.s_kt.append(S_kt)
        
        maxS_kt = np.max(np.cumsum(self.s_kt[::-1]))      
        # print(maxS_kt)  

        if abs(maxS_kt) > self.threshold:
            self.bkp_detect.append(len(self.ts_temp))
            print("CHangeinggggggg")
            buffer = []
            self.ts_temp = []
            self.s_kt = []
            return 10
        return 0    

class FSS(object):
    def __init__(self, signal, mean=[], bkps=[], var=[], fixed_threshold=1500, fixed_size=10, para_known = False):
        #用来逐一存放数据，不更新，主要功能是计/数器
        self.ts = [] 
        #用来逐一存放数据，更新
        self.ts_temp = []
        #用来存放对数似然比
        self.s_kt = []
        #用来存放fixedsize
        self.fixed_size = fixed_size

        self.signal = signal
        
        self.fixed_threshold = fixed_threshold
        #切换点前后是否已知参数
        self.para_known = para_known

        #用来存放真实的切换点
        self.bkps = bkps
        #用来存放真实的均值
        self.mean = mean
        #用来存放真实的方差
        self.var = var
        #用来存放每个数据对应的均值
        self.average = []
        #用来存放切换点的位置
        self.flag = []
        #生成已知的数据平均值
        j=0
        for i in range(len(mean)):
            while j < bkps[i]:
                self.average.append(mean[i])
                j+=1
        
    def add_one(self, ts, one):
        #向列表中添加数据
        ts.append(one)

    def calculate_threshold(self, theta_1, theta_2):
        #计算阈值
        length = len(self.ts_temp)
        threshold = (-2 * (theta_2 - theta_1) * sum(self.ts_temp) + length * (np.square(theta_2) - np.square(theta_1)))
        return threshold

    def calculate_log_likelihood_ratio(self, mean, sig):
        length = len(self.ts_temp)
        log_likelihood_ratio = -2 * abs(sig - mean) * sum(self.ts_temp) + length * abs(np.square(sig) - np.square(mean)) 
        return log_likelihood_ratio

    def sequence_split(self):
        fixed_size_num = len(self.signal) / self.fixed_size -1
        splited_signal = []
        index = 0
        while len(splited_signal) <= fixed_size_num:
            try:
                splited_signal.append(self.signal[index:index+self.fixed_size])
                index += self.fixed_size
            except IndexError:
                splited_signal.append(self.signal[index:])
        return splited_signal
    
    def update(self, x):
        self.add_one(self.ts, x)
        self.add_one(self.ts_temp, x)
        if self.para_known == True:
            try:
                log_likelihood_ratio = self.calculate_log_likelihood_ratio(self.average[len(self.ts)], x)
                # print(self.average[len(self.ts)-1], x)
                self.s_kt.append(log_likelihood_ratio)
            except IndexError:
                self.s_kt.append(0)
        if self.para_known == False:
            try:
                log_likelihood_ratio = self.calculate_log_likelihood_ratio(np.mean(self.ts), x)
                # print(log_likelihood_ratio)
                # print(np.mean(self.ts), x, log_likelihood_ratio)
                self.s_kt.append(log_likelihood_ratio)
            except IndexError:
                self.s_kt.append(0)

    def fss_detection(self):

        splited_signal = self.sequence_split()
        for session in splited_signal:
            for item in session:
                self.update(item)
            if abs(sum(self.s_kt)) >= self.fixed_threshold:
                # print(sum(self.s_kt))
                self.flag.append(1)
                if self.para_known == False:
                    self.ts = []
            else:
                # print(sum(self.s_kt))
                self.flag.append(0)
            self.ts_temp = []
            self.s_kt = []

        # print("FSS window length:"+str(self.fixed_size)+"\n"+"FSS changepoint\n"+str(self.flag.index(1)))
        indicater = []
        for changepoint, flag in enumerate(self.flag):
            if flag == 0:
                indicater.extend([0,0,0,0,0,0])
            else:
                indicater.extend([0,0,0,0,0,12])
                print((changepoint+1) * self.fixed_size)
        return indicater           


if __name__ == '__main__':

    test_data1 = np.load('dataset/Gauss_PRI1.npy')
    test_data2 = np.load('dataset/Gauss_PRI2.npy')
    test_data = np.vstack((test_data1,test_data2))    
    test_data = np.transpose(test_data)
    # plt.subplot(211)
    # plt.plot(test_data1)
    # plt.subplot(212)
    # plt.plot(test_data2)
    # plt.show()

    # D1 = np.load('dataset/D4.npy')
    # X = np.array([D1[:,0]]).T
    # Z = np.array([D1[:,1]]).T
    # scores = []
    bkps = []
    rest = 10
    start = time.time()
#################################################################   CUSUM_BNP-HMM    ###################################################
    CF = CUSUM_BNP_HMM(bkps=[], mean=[], var=[], para_known=False, threshold=1e2)
    for index, sig in enumerate(test_data):
        scor = CF.update(sig)
        if scor > 1:
            bkps.append(index)
            rest = 0
    bkps.append(test_data.shape[0])
    print(bkps)
    exit()


#################################################################   ChangeFinder    ###################################################
    # CF = ChangeFinder(r=0.008, order=10, smooth=7, outlier = False)
    # for index, sig in enumerate(X):
    #     scor, predict = CF.update(sig)
    #     rest += 1
    #     if scor >3 and rest>10:
    #         bkps.append(index)
    #         rest = 0
    #     scores.append(scor)
    # bkps.append(X.shape[0])

#################################################################   CUSUM    ###################################################
    # CF = CUsum(bkps=[], mean=[], var=[], para_known=False, changepoint_th=150) #高斯抖动为28.5
    # for index, sig in enumerate(X):
    #     scor = CF.update(sig)
    #     rest += 1
    #     if scor>1 and rest>10:
    #         bkps.append(index)
    #         rest = 0
    # bkps.append(X.shape[0])

#################################################################   FSS    ###################################################
    CF = FSS(X, bkps=[], mean=[], var=[], para_known=False, fixed_threshold=5000, fixed_size=6)#高斯抖动为800
    indicater = CF.fss_detection()
    for index in range(len(indicater)):
        rest+=1
        if indicater[index]>1 and rest >10:
            bkps.append(index)
            rest = 0
    bkps.append(X.shape[0])

#################################################################   绘图    ###################################################
    print("time cosuming:" + str(time.time()-start))
    fig = plt.figure(dpi=300,figsize=(16,8))
    plt.xlabel("PRI index")
    plt.ylabel("PRI value")
    # plt.title("stagger PRI")
    bank = ['r','g','b','orange','brown','purple','rosybrown','black']
    plt.plot(range(bkps[0]),X[:bkps[0]],marker="*", color=bank[0])
    for index in range(1,len(bkps)):
        plt.plot(range(bkps[index-1]+1, bkps[index]),X[bkps[index-1]+1: bkps[index]], marker="*", color=bank[index])

    plt.savefig("draw/FSS/D4.pdf",dpi=300)
    plt.show()
# %%
