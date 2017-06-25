from scipy.io.wavfile import read
import numpy as np
from math import *
from scipy.fftpack import dct
def preDeal(data):
    result = np.zeros(len(data))
    result[0] = data[0]
    for n in range(1,len(data)):
        result[n] = data[n]-0.95*data[n-1]

    return result

#分帧,加窗及补零，默认采样频率为16000
def split(data):
    fs = 16000
    length = 0.025*fs   #25ms所占据的帧长度
    step = 0.01*fs      #10ms采样步长，
    row = len(data)//step-3
    result = np.zeros((int(row),512))
    zero = np.zeros(112)
    for i in range(int(row)):
        tmp = np.zeros(400)
        for j in range(int(length)):
            tmp[j] = (0.54-0.46*cos(2*pi*j/length))*data[int(step*row+j)]  #加窗
        result[i] = np.append(tmp,zero)   #补零至512便于做FFT
    return result,(row,512)

#做FFT并频谱取模的平方得到功率谱
def frePower(slides,info):
    Fs = 16000
    result = np.zeros((int(info[0]),int(info[1]//2)))
    for i in range(int(info[0])):
        n = len(slides[i])
        Y = np.fft.fft(slides[i])/n         #进行FFT
        result[i] = np.square(np.absolute(Y[range(n//2)]))  #取模的平方

    return result,(info[0],info[1]//2)

#定义单个梅尔滤波器的输出
def mel(k,i,f):
    if (k<f[i-1]) or (k>f[i+1]):
        result = 0
    elif f[i-1]<=k and k<=f[i]:
        result = (k-f[i-1])/(f[i]-f[i-1])
    else:
        result = (f[i+1]-k)/(f[i+1]-f[i])
    return result
#进行梅尔滤波,并取对数
def melFilter(slides,info):
    Mf = 1125*log(1+8000/700,np.e)          #求梅尔变换后最大的频率
    f = np.zeros(40)
    fs = 16000
    for i in range(40):
        f[i] = Mf/40*i
        f[i] = 700*(np.exp(f[i]/1125)-1)    #构造f频率数组，记录第i个滤波器的中心频率
    for i in range(int(info[0])):
        for j in range(info[1]):
            result =0
            for cnt in range(39):
                result = result+slides[i][j]*mel(j/info[1]*fs,cnt,f)
            if result!=0:
                slides[i][j] = log(result,10)
            else:
                slides[i][j] = 0
    return slides,info

#DCT变换并归一化
def dctNomalize(slides,info):
    result = np.zeros((int(info[0]),13))
    for i in range(int(info[0])):
        slides[i] = dct(slides[i])             #对数据进行DCT变换
        for _ in range(13):
            result[i][_] = slides[i][_]     #取前13个数据
        result[i] = (result[i]-np.mean(result[i]))/np.std(result[i])   #归一化
    return result

#定义MFCC算法
def mfcc(data):
    preData = preDeal(data) #预加重处理
    slides,info1 = split(preData) #分帧、加窗并补零
    fP,info2 = frePower(slides,info1) #计算FFT,及其功率谱
    LE,info3 = melFilter(fP,info2)  #进行梅尔滤波并取对数
    result = dctNomalize(LE,info3)  #进行DCT变换得到梅尔倒谱
    return result,(info3[0],13)

#进行差分运算
def dv(a,n):
    result = np.zeros(n-1)
    result1 = np.zeros(n-2)
    for i in range(n-1):
        result[i] = a[i+1]-a[i]
    for j in range(n-2):
        result1[j] = result[j+1]-result[j]
    return result

#对本人进行测试
Fs, data=read('./enroll/F101_train.wav')
sample,info = mfcc(data)
for k in range(1,4):
    testName = './test/F101_test_'+str(k)+'.wav'
    Fs, data = read(testName)
    test, info2 = mfcc(data)

    result = 0
    result2 = 0
    result3 = 0
    for i in range(900):
        sample[2 * i] = sample[2 * i] / sqrt(np.sum(np.square(sample[2 * i])))
        test[i] = test[i] / sqrt(np.sum(np.square(test[i])))
        s = dv(sample[2 * i], 13)
        t = dv(test[i], 13)
        # s = s/sqrt(np.sum(np.square(s)))
        # t = t/sqrt(np.sum(np.square(t)))
        result = result + sample[2 * i].dot(test[i])
        result2 = result2 + s.dot(t)
        #result3 =result3+np.sum(np.square(s-t))
    result = result / 900
    result2 = result2 / 900
    #result3 = result3/900
    print(result, result2,result3)

print("**********************************************************")
#对他人进行测试
for i in range(46,88):
    testName = './test/F1'+u'%02d' %i+'_test_'+str(1)+'.wav'
    Fs, data=read(testName)
    test,info2 = mfcc(data)

    result = 0
    result2 =0
    for i in range(900):
        sample[2*i] = sample[2*i]/sqrt(np.sum(np.square(sample[2*i])))
        test[i] = test[i]/sqrt(np.sum(np.square(test[i])))
        s = dv(sample[2*i],13)
        t = dv(test[i],13)
        #s = s/sqrt(np.sum(np.square(s)))
        #t = t/sqrt(np.sum(np.square(t)))
        result = result+sample[2*i].dot(test[i])
        result2 = result2+s.dot(t)
        #result3 = result3 + np.sum(np.square(s - t))
    result = result/900
    result2 = result2/900
    #result3 = result3/900
    print(result,result2,result3)
















