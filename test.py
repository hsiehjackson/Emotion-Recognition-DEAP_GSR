from Ledapy.runner import *
import scipy.io as sio
from numpy import array as npa
import matplotlib.pyplot as plt

filename = 'Ledapy/EDA1_long_100Hz.mat'
sampling_rate = 100
downsample = 4
matdata = sio.loadmat(filename)
rawdata = npa(matdata['data']['conductance'][0][0][0], dtype='float64')/1e6
time_data = utils.genTimeVector(rawdata, sampling_rate)
(time_data, conductance_data) = utils.downsamp(time_data, rawdata, downsample, 'mean')
phasicdata = getResult(rawdata, 'phasicdata', sampling_rate, downsample=downsample, optimisation=2)
tonicdata = conductance_data - phasicdata
plt.plot(conductance_data,'b')
plt.plot(tonicdata,'g')
plt.show()