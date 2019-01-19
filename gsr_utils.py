import numpy as np
from scipy.signal import butter, filtfilt, detrend, argrelextrema, resample, sosfilt
from biosppy.signals import eda
from biosppy.signals import tools
import matplotlib.pyplot as plt
from pyentrp import entropy as ent
import itertools
from Ledapy.runner import *
from Ledapy.cvxeda import *


def rename(feature_name,process_name):
	return [process_name+name for name in feature_name]

"==========preprocess=========="

def delpeak(signal,stdnum,iter):
	data = signal
	for i in range(iter):
		data = np.delete(data,np.where(abs(data-np.mean(data))>(stdnum*np.std(data))))
		pos = np.where(abs(np.diff(data)-np.mean(np.diff(data)))>stdnum*np.std(np.diff(data)))[0]
		pos = np.unique(np.concatenate((pos,pos+1,pos-1)))
		data = np.delete(data, pos)
	return data

def normalize(data):
	mean = np.mean(data)
	std = np.std(data)
	data = (data-mean)/std
	return data

def low_pass_filter(data, fc, fs=128, order=5):
	nyq = 0.5 * fs
	normal_cutoff = fc / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	y = filtfilt(b, a, data)
	return y 

def high_pass_filter(data, fc, fs=128, order=5):
	nyq = 0.5 * fs
	normal_cutoff = fc / nyq
	b, a = butter(order, normal_cutoff, btype='high', analog=False)
	y = filtfilt(b, a, data)
	return y 

def band_pass_filter(data, lowcut, highcut, fs=128, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band',analog=False)
	y = filtfilt(b, a, data)
	return y

def downsample(data, fs, nfs):
	time_data = utils.genTimeVector(conductance=data, srate=fs)
	time_data, SC = utils.downsamp(t=time_data, data=data, fac=int(fs/nfs), method='mean')
	return SC


"==========SCR=========="
def CDASCR(data,fs,min_amplitude):
	CDAdriver = getResult(raw_vector=data, result_type='phasicdriver', sampling_rate=fs, downsample=8, optimisation=2, pipeout=None)
	CDAphasic = getResult(raw_vector=data, result_type='phasicdata', sampling_rate=fs, downsample=8, optimisation=2, pipeout=None)
	onsets, pks, amps = find_zeropeak(CDAdriver, min_amplitude, 'CDA')
	return onsets, pks, amps, CDAdriver, CDAphasic, 
def CVXSCR(data,fs,min_amplitude):
	tonic, phasic = cvxEDA(data,sampling_rate=fs)
	tonic = np.array(tonic)[:,0]
	phasic = np.array(phasic)
	onsets, pks, amps = find_zeropeak(phasic, min_amplitude, 'CVX')
	return onsets, pks, amps, phasic, tonic
def detrendSCR(data,fs,min_amplitude):
	#onset at local minimum; peak at local maximum
	gsr_detrend = detrend(data)
	gsr_detrend = gsr_detrend+abs(np.min(gsr_detrend))
	#plotsignal([data,gsr_detrend],False)
	onsets, pks, amps = find_zeropeak(gsr_detrend,min_amplitude,'detrend')
	return onsets, pks, amps, gsr_detrend
def windowSCR(data,fs,min_amplitude):
	#onset at mean==median; peak at median>mean
	def meanfilter(data, fs):
		def mean(lst): return sum(lst)/len(lst)
		env = np.zeros_like(data).astype('float')
		for i in range(len(data)):
			env[i] = mean(data[max(int(i-fs/2+1),0):int(i+1+fs/2)])
		return env
	gsr_avg = meanfilter(data,fs*3)
	gsr_scr = data - gsr_avg
	gsr_scr = gsr_scr - np.min(gsr_scr)
	#plotsignal([data,gsr_scr],False)
	onsets, pks, amps = find_zeropeak(gsr_scr,min_amplitude,'window')
	return onsets,pks,amps,gsr_scr
def diffSCR(data,fs,min_amplitude):
 	#onset at local minimum; peak at diff maximum
	df = np.diff(data)
	df = np.append(df,df[-1])
	size = int(1. * fs)
	df = tools.smoother(df,'bartlett',size)['signal']
	#plotsignal([data,df],False)
	onsets, pks, amps = find_zeropeak(df,min_amplitude,'diff')
	return onsets,pks,amps,df

def freqSCR(data,feq,type_filter,fs,min_amplitude):
	if type_filter == 'low':
		tonic = low_pass_filter(data, fc=feq[0],fs=fs, order=5)
		gsr_filter = data - tonic #phasic
		gsr_filter = gsr_filter - np.min(gsr_filter)
		onsets, pks, amps = find_zeropeak(gsr_filter,min_amplitude,'filter')
	elif type_filter == 'band':
		gsr_filter = band_pass_filter(data, lowcut=feq[0], highcut=feq[1], fs=fs, order=5)
		gsr_filter = gsr_filter - np.min(gsr_filter)
		onsets, pks, amps = find_zeropeak(gsr_filter,min_amplitude,'filter')
	elif type_filter == 'high':
		gsr_filter = high_pass_filter(data, fc=feq[0],fs=fs, order=5)
		onsets, pks, amps = find_zeropeak(gsr_filter,min_amplitude,'filter')
	
	return onsets,pks,amps,gsr_filter

def find_zeropeak(data,min_amplitude,task):	
	scrs, amps, ZC, pks = [], [], [], []
	if task=='diff': 
		zeros, = tools.zero_cross(signal=data, detrend=False)
		lm1 = argrelextrema(data[:zeros[0]],np.less)[0]
		lm2 = argrelextrema(data[zeros[-1]:],np.less)[0]+zeros[-1]
		if len(lm2)!=0:
			if zeros[-1]!=lm2[-1]:
				zeros = np.insert(zeros,len(zeros),lm2[-1])
			elif zeros[-1]!=len(data)-1:
				zeros = np.insert(zeros,len(zeros),len(data)-1)
		elif zeros[-1]!=len(data)-1:
			zeros = np.insert(zeros,len(zeros),len(data)-1)
		if len(lm1)!=0:
			if zeros[0]!=lm1[0]:
				zeros = np.insert(zeros,0,lm1[0])
			elif zeros[0]!=0:
				zeros = np.insert(zeros,0,0)				
		elif zeros[0]!=0:
			zeros = np.insert(zeros,0,0)
	else:
		zeros = argrelextrema(data,np.less)[0]
		zeros = np.insert(zeros,len(zeros),len(data)-1)
		zeros = np.insert(zeros,0,0)
	'''
	ts = np.linspace(0, (len(data)-1)/20, len(data),endpoint=False)
	plt.scatter(ts,data,s=2)
	plt.scatter(ts[zeros],data[zeros],c='y',s=10)
	plt.show()
	'''
	for i in range(0, len(zeros) - 1, 1):
		scrs += [data[zeros[i]:zeros[i + 1]+1]]
		aux = scrs[-1].max()
		#print(aux, data[zeros[i]], data[zeros[i+1]])
		if aux > data[zeros[i]] and aux > data[zeros[i+1]]:
			#print(aux)
			amps += [aux-data[zeros[i]]]
			ZC += [zeros[i]]
			ZC += [zeros[i + 1]]
			pks += [zeros[i] + np.argmax(data[zeros[i]:zeros[i + 1]])]
		elif aux == data[zeros[-1]]:
			amps += [aux-data[zeros[-2]]]
			ZC += [zeros[-2]]
			ZC += [zeros[-1]]
			pks += [zeros[-1]]
	if amps == []:
		ZC += [np.argmin(data)]
		amps += [np.max(data[ZC[0]:])-data[ZC[0]]]
		pks += [np.argmax(data[ZC[0]:])]

	scrs = np.array(scrs)
	amps = np.array(amps)
	ZC = np.array(ZC)
	pks = np.array(pks)
	onsets = ZC[::2]
	thr = min_amplitude * np.max(amps)
	arglow = np.where(amps<thr)
	amps = np.delete(amps,arglow)
	pks = np.delete(pks,arglow)
	onsets = np.delete(onsets,arglow)

	risingtimes = pks-onsets
	risingtimes = risingtimes/16

	pks = pks[risingtimes > 0.1]
	onsets = onsets[risingtimes > 0.1]
	amps = amps[risingtimes > 0.1]
	return onsets,pks,amps

"=========frequency========="
def band2idx(freq, cutoff_low, cutoff_high):
	index = []
	for i in range(len(freq)):
		if freq[i] < cutoff_high and freq[i] >= cutoff_low:
			index.append(i)
	return np.array(index)

def sef(feq, power, ratio):
	for i, f in enumerate(feq):
		if np.sum(power[0:i]) > (np.sum(power)*ratio):
			return (feq[i]*(np.sum(power)*ratio-np.sum(power[0:i-1]))+feq[i-1]*(np.sum(power[0:i])-np.sum(power)*ratio))/power[i]

def get_freq_info(power,feq):
	max_freq = feq[np.argmax(power)]
	min_freq = feq[np.argmin(power)]
	mean_freq = np.inner(feq,power)/np.sum(power)
	median_freq = sef(feq, power, 0.5)
	q1_freq = sef(feq,power, 0.25)
	q3_freq = sef(feq,power, 0.75)
	IR_freq = q3_freq - q1_freq	
	feature_name = ['mxf','mif','mef','mdf','IRf']	
	feature = [max_freq, min_freq, mean_freq, median_freq, IR_freq]
	return feature, feature_name
"==========entropy=========="

def information_entropy(data,match):
	entropy_data = np.zeros(len(data)-1)
	data = normalize(data)
	diff = np.diff(data)
	std = np.std(diff)
	for i in range(len(diff)):
		if diff[i] >= 0 and abs(diff[i]) < std:
			entropy_data[i] = 0
		elif diff[i] >= 0 and abs(diff[i]) >= std:
			entropy_data[i] = 1
		elif diff[i] < 0 and abs(diff[i]) >= std:
			entropy_data[i] = 2
		elif diff[i] < 0 and abs(diff[i]) < std:
			entropy_data[i] = 3
	entropy_vector = []
	for i in range(len(entropy_data)-match+1):
		vector = []
		for j in range(int(match)):
			vector.append(entropy_data[i+j])
		entropy_vector.append(vector) 
	dictprob = {}
	for i in entropy_vector:
		collect = tuple(i)
		if collect not in dictprob:
			dictprob[collect] = 1
		else:
			dictprob[collect]+=1
	prob = np.array(list(dictprob.values())).astype('float')
	prob = prob/np.sum(prob)
	entropy = 0
	for i in range(len(prob)):
		entropy += -1*prob[i]*np.log2(prob[i])
	return entropy

def ap_entropy(X, match, tolerance):
	def embed_seq(X, Tau, D):
		shape = (X.size - Tau * (D - 1), D)
		newX = np.zeros((X.size-Tau*(D-1),D))
		for i in range(len(newX)):
			for j in range(D):
				newX[i][j] = X[i+Tau*j]
		return newX
	def getCM(X,M,R):
		N = len(X)
		Em = embed_seq(X, 1, M)
		A = np.tile(Em, (len(Em), 1, 1))
		B = np.transpose(A, [1, 0, 2])
		D = np.abs(A - B) #  D[i,j,k] = |Em[i][k] - Em[j][k]| # with value to every j
		InRange = np.max(D, axis=2) <= R
		Cm = InRange.mean(axis=0) #  Probability that random M-sequences are in range
		print(np.sum(np.log(cm)))
		input()
		phi = np.sum(np.log(Cm))/(N-M+1) 
		return phi
	Phi_m = getCM(X,match,tolerance)
	Phi_mp = getCM(X,match+1,tolerance)
	Ap_En = (Phi_m - Phi_mp)
	return Ap_En

def coarse_grain(time_series, scale):
	allList = []
	for s in range(scale):
		b = np.fix((len(time_series)-s)/scale)
		oneList = []
		for i in range(0,int(b*scale),scale):
			oneList.append(np.mean(time_series[s+i:s+i+scale]))
		allList.append(oneList)
	return allList

def samp_ent(time_series, m, tolerance=None, entropy=False):
	def embed_seq(X, Tau, D):
		shape = (len(X) - Tau * (D - 1), D)
		newX = np.zeros((len(X)-Tau*(D-1),D))
		for i in range(len(newX)):
			for j in range(D):
				newX[i][j] = X[i+Tau*j]
		return newX
	def getCM(X,M,R):
		N = len(X)
		Em = embed_seq(X, 1, M)
		A = np.tile(Em, (len(Em), 1, 1))
		B = np.transpose(A, [1, 0, 2])
		D = np.abs(A - B)
		InRange = np.max(D, axis=2) <= R
		sumCm = InRange.sum()
		Cm = (sumCm - len(Em))/2
		return Cm
	if entropy==True:
		nM = getCM(time_series,m,tolerance)
		nM_1 = getCM(time_series,m+1,tolerance)
		return -np.log(nM_1/nM)
	else:
		return getCM(time_series,m,tolerance)

def perm_ent(time_series, m, delay, entropy=False):
	def util_hash_term(perm):
		deg = len(perm)
		return sum([perm[k]*deg**k for k in range(deg)])

	n = len(time_series)
	permutations = np.array(list(itertools.permutations(range(m))))

	hashlist = [util_hash_term(perm) for perm in permutations]
	c = [0] * len(permutations)
	for i in range(n - delay * (m - 1)):
		sorted_index_array = np.array(np.argsort(time_series[i:i+delay*m:delay], kind='quicksort'))
		hashvalue = util_hash_term(sorted_index_array);
		c[np.argwhere(hashlist == hashvalue)[0][0]] += 1
	c = [element for element in c if element != 0]
	p = np.divide(np.array(c), float(sum(c)))
	if entropy == True:
		return -sum(p * np.log(p))	
	else:
		return p

def RCMSE(time_series, match, scale, tolerance):
	taulist = coarse_grain(time_series,scale)
	nM = []
	nM_1 = []
	for t in taulist:
		t = np.array(t)
		nM.append(samp_ent(t,match,tolerance))
		nM_1.append(samp_ent(t,match+1,tolerance))
	nM = np.mean(np.array(nM))
	nM_1 = np.mean(np.array(nM_1))
	return -np.log(nM_1/nM)

def RCMPE(time_series, match, scale, delay):
	taulist = np.array(coarse_grain(time_series,scale))
	p = []
	for t in taulist:
		p.append(perm_ent(t,match,delay))
	p = np.mean(np.array(p),axis=0)
	return -np.sum(p * np.log(p))

"==========plot=========="

def plotsignal(pic,save=False,filename=None):
	for i in range(len(pic)):
		plt.subplot(len(pic),1,i+1)
		plt.plot(pic[i])
	if save==True:
		plt.savefig(filename)
		plt.close()
	else:
		plt.show()
		plt.close()

def plotpeak(orign,pksig,tonic,onset,peak,amp,filename):
	#tonic = orign-(pksig-np.min(pksig)) #because negative
	plt.subplot(2,1,1)
	ts = np.linspace(0, (len(pksig)-1)/16,len(pksig),endpoint=False)
	plt.scatter(ts,pksig,s=2)
	plt.scatter(ts[peak],pksig[peak],c='r',s=10)
	plt.scatter(ts[onset],pksig[onset],c='y',s=10)
	plt.subplot(2,1,2)
	ts = np.linspace(0, (len(orign)-1)/16,len(orign),endpoint=False)
	plt.scatter(ts,orign,s=2,c='b')
	plt.scatter(ts,tonic,s=2,c='g')
	plt.scatter(ts[peak],orign[peak],c='r',s=20)
	plt.scatter(ts[onset],orign[onset],c='y',s=10)
	plt.show()
	plt.close()


