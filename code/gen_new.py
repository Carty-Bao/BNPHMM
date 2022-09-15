
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 21:42:00 2021

@author: fyy
"""
import scipy.stats as stats
import numpy as np 
import random
import scipy.io as scio
import matplotlib.pyplot as plt
import math



dataFile = './_dat/val_dataset.mat'
ratio = 0.05
sample_num = 100 # 训练样本的大小
max_len = 250
min_len = 180
max_kn = 4
min_kn = 0
s_length = 200



def stable(maxLen,priValue):
    priSeq = np.ones(maxLen)*priValue
    return priSeq


def jitter(maxLen,priValue,priDev):

    maxDevValue = priValue*priDev
    lowerBound = priValue-maxDevValue
    upperBound = priValue+maxDevValue
    priSeq = np.random.randint(lowerBound,upperBound+1,maxLen)#lower<=x<upper
    params = [priValue,priDev,maxDevValue]
    return priSeq


#周期
def periodic(maxLen,priValue,ratio):

    amp=priValue*ratio; #正弦幅度
    freq=50; #正弦频率
    sample =  random.randint(2*freq,8*freq) #正弦采样率
    fsample=400#正弦采样率
    
    priDSSeq = np.zeros(maxLen)
    for i in range(maxLen):
        priDSSeq[i] = amp*math.sin(freq*i/fsample)+priValue#正弦PRI序列
    #priDSSeq = priDSSeq[:maxLen] #截断
    para = [priValue,ratio,sample]
    return priDSSeq

#滑变
def sliding(maxLen,priValue,ratio):
    priMax=priValue*ratio #pri最大值
    pulseNum=random.randint(ratio,32) #pri点数
    slidingStep=(priMax-priValue)/pulseNum; #滑变步长
    slidingSeq = np.zeros(pulseNum+1)
    for i in range(pulseNum+1):
        #一个周期的滑变PRI序列
        slidingSeq[i] =  i*slidingStep + priValue
    seqLen=len(slidingSeq);
    
    cycleNum=math.ceil(maxLen/seqLen)#向上取整周期数
    priDSSeq = np.tile(slidingSeq, cycleNum)#重复若干个周期
    priDSSeq = priDSSeq[:maxLen] #截断
    para = [priValue,ratio,priMax,pulseNum,slidingStep]
    return priDSSeq
'''
import numpy as np
a = np.array([[1, 2, 0, 3, 0],
       [4, 5, 0, 6, 0],
       [7, 8, 0, 9, 0]])
 
idx = np.argwhere(np.all(a[..., :] == 0, axis=0))
a2 = np.delete(a, idx, axis=1)
'''
#参差 3-10
def stagger(maxLen,priValue,priNum):
    seqLen=priNum #一个周期的脉组中脉冲数目
    cycleNum=math.ceil(maxLen/seqLen) #周期数
    
    priSeq = priValue
    priSSeq=np.tile(priSeq,cycleNum)#重复若干周期
    priSSeq=priSSeq[:maxLen]#截断                                                                                                                
    para = [priValue,priNum,cycleNum]
    return priSSeq
def gen_func(m,maxLen):
    if m==1:
        return stable(maxLen)
    elif m==2:
        return jitter(maxLen)
    elif m==3:
        return periodic(maxLen)
    elif m==4:
        return sliding(maxLen)
    elif m==5:
        return stagger(maxLen)
    else:
        print("****error!****")

def solve(nums, x, y) :
   if nums == []:
       return False
   if x>y:
       (x,y) = (y,x)
   for i in range(len(nums)):
       if x<= nums[i] <= y:
           return True
       else:
           continue
   return False


def pri2toa(inputseq):
    #mask = np.logical_not(inputseq)
    mask = inputseq!=0
    inputseq = inputseq[mask]
    toa = np.zeros(len(inputseq)+1)
    i = 0
    while(i<len(inputseq)):
        toa[i+1] = toa[i]+inputseq[i]
        i = i+1
    return toa

max_len = 250

def lostPul(inputseq,proportion,label,pattern):#缺失脉冲
    #   inputseq:   输入TOA序列
    #   proportion: 缺失百分比
    #   seqTOA:     缺失的TOA序列
    #   seqPRI:     缺失的PRI序列
    lostPulseSeq=pri2toa(inputseq) #每个proportion下面的缺失脉冲TOA序列
    lengthWorkModeSample=len(lostPulseSeq)
    rand_num = math.floor(lengthWorkModeSample*proportion)
    randomIndex=np.random.randint(0,lengthWorkModeSample,rand_num)#lower<=x<upper
    randomIndex = sorted(randomIndex)
    
    j=0
    mask = label!=0
    label = label[mask]
    lostlabel = label*1 #单纯a = b 只是浅复制将a指向b
    p = pattern*1
    for i in range(len(randomIndex)):
        while(j<len(label) and randomIndex[i]>=label[j]):
            j = j+1
        lostlabel[j:] = lostlabel[j:] - 1
    
    lostPulseSeq=[i for num,i in enumerate(lostPulseSeq) if num not in randomIndex]
    p =[i for num,i in enumerate(p) if num not in randomIndex]
    p = np.array(p)
    for i in range(len(randomIndex)):
        p[randomIndex[i]-1-i] = 6
    lostPulseSeq = np.array(lostPulseSeq)
    seqPRI=lostPulseSeq[1:]-lostPulseSeq[:-1]
    seqTOA=lostPulseSeq
    
    z = np.zeros(max_len)
    seqPRI = np.append(seqPRI,z)
    p = np.append(p,z)
    
    z = np.zeros(5)
    lostlabel = np.append(lostlabel,z)
    return seqPRI[:max_len],lostlabel[:5],p[:max_len]

def findpos(arr,x):
    for i in range(len(arr)):
        if arr[i]>x:
            return i
    return -1
def suprPul(inputseq,proportion,label,p):#虚警脉冲
    #   inputseq:   输入TOA序列
    #   proportion: 虚警百分比
    #   seqTOA:     虚警的TOA序列
    #   seqPRI:     虚警的PRI序列
    #   pw:         脉宽,脉冲串脉宽设置为5us
    supPulseSeq=pri2toa(inputseq) #每个proportion下面的缺失脉冲TOA序列
    lengthWorkModeSample=len(supPulseSeq)
    tMax = math.floor(max(supPulseSeq))
    randomNum = math.floor(lengthWorkModeSample*proportion)
    randomTime=np.random.randint(0,tMax,randomNum)
    randomTime = sorted(randomTime)
    
    mask = label!=0
    label = label[mask]
    pattern = p*1
    j = 0
    for i in range(len(randomTime)):
        pos = findpos(supPulseSeq,randomTime[i])
        while(j<len(label) and label[j] < pos):
            j = j+1
        label[j:] = label[j:] + 1
        supPulseSeq = np.insert(supPulseSeq, pos,randomTime[i])
        pattern[pos-1] = 6
        pattern = np.insert(pattern, pos,6)
        
    randomIndex=[i for i,val in enumerate(supPulseSeq) if val in randomTime]
    seqPRI=supPulseSeq[1:]-supPulseSeq[:-1]
    seqTOA=supPulseSeq
    
    z = np.zeros(max_len)
    seqPRI = np.append(seqPRI,z)
    z = np.zeros(5)
    label = np.append(label,z)
    return seqPRI[:max_len],label[:5],pattern[:max_len]

def meaErr(inputseq,stdPRI):
#   inputseq:   输入TOA序列
#   stdPRI:     测量误差的标准差
#   seqTOA：    输出TOA序列
#   seqPRI：    输出PRI序列 
    seqTOA=pri2toa(inputseq)
    lengthWorkModeSample=len(seqTOA)
    errGenarated = np.random.normal(0, stdPRI, lengthWorkModeSample)
    #errGenarated=normrnd(0,stdPRI,[1,lengthWorkModeSample])
    seqTOA=seqTOA+errGenarated
    seqPRI=seqTOA[1:]-seqTOA[:-1]
    return seqPRI[:max_len]

def indices(a,func):
    #实现find函数
    return [i for (i,val) in enumerate(a) if func(val)]
#a = [1 2 3 1 2 3]
#find = indices(a,lambda x:x>2) --> [2,5]


data = np.zeros((sample_num, max_len), dtype=np.float32)
label = np.zeros((sample_num, max_kn+1), dtype=np.int)
pattern = np.zeros((sample_num, max_kn+1), dtype=np.int)
p = np.zeros((sample_num, max_len), dtype=np.float32)

for i in range(sample_num):
   #seq_len = random.randint(min_len,max_len)
   seq_len = max_len
   knum = random.randint(min_kn,max_kn)
   k = []
   for j in range(knum):
       a = random.randint(25,s_length-25)
       while solve(k,a-25,a+25):
           a = random.randint(25,s_length-25)
       k.append(a)
   k.append(seq_len)
   k = np.array(k)
   k = sorted(k)
   priValue = random.randint(10,20)*10
   priDev = random.randint(10,20)/20
   
   for j in range(knum+1):
       label[i,j] = k[j]
       module = 2
       pattern[i,j] = module
       flag = random.randint(1,3)
       tempValue = priValue
       tempDev = priDev
       if flag == 1:#均值方差全变
           while(tempValue==priValue):
               tempValue = random.randint(10,20)*10
               
           while(tempDev==priDev):
               tempDev = random.randint(10,20)/20

       elif flag == 2:#只变均值
           while(tempValue==priValue):
               tempValue = random.randint(10,20)*10
       else:#只变均值
           while(tempDev==priDev):
               tempDev = random.randint(10,20)/20
               
       priValue = tempValue
       priDev = tempDev
       if j==0:
           data[i,:k[j]] = jitter(k[j],priValue,priDev)
           p[i,:k[j]] = module
       else:
           data[i,k[j-1]:k[j]] = jitter(k[j]-k[j-1],priValue,priDev)
           p[i,k[j-1]:k[j]] = module
           

d = data*1
l = label*1
result = np.zeros((sample_num, s_length), dtype=np.float32)
L = np.zeros((sample_num, s_length), dtype=np.float32)

'''  
for i in range(sample_num):
    d[i]  =meaErr(data[i],1)

for i in range(sample_num):
    d[i],l[i],p[i]  = lostPul(data[i],0.1,l[i],p[i])#247.5
  
for i in range(sample_num):
    d[i],l[i],p[i]  = suprPul(data[i],0.05,l[i],p[i])#247.5

'''

d = d[:,:s_length]
p = p[:,:s_length]

for i in range(sample_num):
    for j in range(max_kn+1):
        if l[i,j]>=s_length:
            l[i,j:] = 0
            l[i,j] = s_length
            break

for i in range(sample_num):
    for j in range(max_kn+1):
        if l[i,j] != s_length and l[i,j] != 0:
            result[i,l[i,j]] = 1
            L[i,l[i,j]] = 1
            result[i,l[i,j]-1] = 0.8
            result[i,l[i,j]+1] = 0.8

plt.plot(d[0])

# scio.savemat(dataFile, {'data':d,'label':result,'pattern':p,'L':L,'Y':d,'l_true':l})

