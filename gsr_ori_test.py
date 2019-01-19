import sys 
import csv
import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
from gsr_utils import *


def downsample(data, fs, nfs):
	secs = float(len(data)/fs)
	sample = int(secs*nfs)
	newdata =  resample(data, sample)
	return newdata

dataname = sys.argv[1]
file = open(dataname,'r')
signal = []
file.readline()
for n, line in enumerate(csv.reader(file)):
	print('\r{} {} {}'.format(n+1, line[41], line[42]),end='')
	signal.append(line[41])
signal = np.array(signal).astype('float')
signal = downsample(signal,512,128)
print(signal.shape)
for i in range(40):
	#signal = signal[1000:9064]
	#signal = signal - np.mean(signal[1000:1384])
	#sig = signal[16512+i*8064:16512+8064+i*8064]
	#sig = signal[29568+i*8064:29568+8064+i*8064]
	sig = signal[40704+i*8064:40704+8064+i*8064]
	#mean = np.mean(sig[:384])
	#std = np.std(sig[:384])
	#sig = (sig - mean)*std*10
	plt.plot(sig)#132-192

	#plt.plot(signal[29824+i*8064:29824+8064+i*8064])#233-294
	#plt.plot(signal[41088+i*8064:41088+8064+i*8064])#321-381
	plt.show()


