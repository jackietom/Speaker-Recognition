import matplotlib.pyplot as plt
import plotly.plotly as py
import numpy as np

#对声音进行FFT
def audioFft(Fs, y):
    Ts = 1.0/Fs; # sampling interval
    yFft = np.zeros((25*len(y)//(Fs),Fs//50))
    sample = np.split(y,25*len(y)//(Fs))
    i = 0
    for s in sample:
        n = len(s) # length of the signal
        Y = np.fft.fft(s)/n # fft computing and normalization
        Y = Y[range(n//2)]
        yFft[i] = Y
        i = i+1

    result = np.zeros(Fs//50)
    for cnt in range(Fs//50):
        for cnt2 in range(25*len(y)//(Fs)):
            result[cnt] = result[cnt]+yFft[cnt2][cnt]
    result = result/(25*len(y)//(Fs))
    return result

