from fft import audioFft
from lowpassFilter import lowpassDeal
from scipy.io.wavfile import read
import numpy as np
from math import *

#fn = os.path.abspath()+'\data_thuyg20_sre\enroll\F101_train.wav'
#print(fn)
#读取数据，滤波做FFT后返回
def enrollData(filename):
    Fs, data=read(filename)
    data = lowpassDeal(Fs,data)
    fftResult=audioFft(Fs,data)
    fftResult2 = abs(fftResult)
    s=np.sum(np.square(fftResult2))
    fftResult2 = fftResult2/sqrt(np.sum(np.square(fftResult2)))
    s2=np.sum(np.square(fftResult2))
    return fftResult2

def TestData(filename):
    Fs, data = read(filename)
    data = lowpassDeal(Fs, data)
    fftResult = audioFft(Fs, data)
    fftResult2 = abs(fftResult)
    fftResult2 = fftResult2 / sqrt(np.sum(np.square(fftResult2)))
    return fftResult2

对本人进行测试
for i in range(1,18):
    enroll = './enroll/F101_train.wav'
    test = './test/F101_test_'+str(i)+'.wav'

    enrollResult = enrollData(enroll)
    testResult = TestData(test)
    print(abs(enrollResult.dot(testResult)))
print('**************************************')
rst = 0
#对他人进行测试
for i in range(46,88):
    enroll = './enroll/F101_train.wav'
    test = './test/F1'+u'%02d' %i+'_test_'+str(1)+'.wav'
    enrollResult = enrollData(enroll)
    testResult = TestData(test)
    a=enrollResult.dot(testResult)
    print(a)
    rst = rst+a
rst = rst/42
print(rst)

