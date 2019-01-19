import numpy as np
from scipy.signal import butter, filtfilt, detrend, argrelextrema, resample, welch, stft, periodogram
from biosppy.signals import eda
from biosppy.signals import tools
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import pywt
from gsr_utils import *
import math



def statistics_feature(data):
	mean = np.mean(data)
	std = np.std(data)
	Skew = skew(data)
	kurt = kurtosis(data)
	mean_fst_absdiff = np.mean(abs(np.diff(data)))
	mean_snd_absdiff = np.mean(abs(np.diff(np.diff(data))))
	mean_fst_diff = np.mean(np.diff(data))
	mean_snd_diff = np.mean(np.diff(np.diff(data)))
	mean_neg_diff = np.mean(np.diff(data)[np.where(np.diff(data)<0)])
	proportion_neg_diff = len(np.where(np.diff(data)<0)[0])/(len(np.diff(data)))
	number_local_min = len(argrelextrema(data,np.less)[0])
	number_local_max = len(argrelextrema(data,np.greater)[0])
	f1 = [mean,std,Skew,kurt]
	f2 = [mean_fst_absdiff,mean_snd_absdiff,mean_fst_diff,mean_snd_diff,mean_neg_diff,proportion_neg_diff]
	f3 = [number_local_min,number_local_max]
	f = f1+f2+f3
	name = ['me','std','sk','ku','me.1absdf','me.2absdf','me.1df','me.2df','me.negdf','ro.negdf','num.argmi','num.argma']
	return f, name

def SCR_feature(data,onsets,peaks,amps,fs):
	maxSCR = np.max(amps)
	minSCR = np.min(amps)
	avgSCR = np.mean(amps)
	freqSCR = len(amps)/len(data)
	durSCR = np.mean(np.diff(peaks))/fs if len(peaks)!=1 else peaks[0]/fs
	
	avg_riseT = np.mean(peaks-onsets)/fs
	avg_recovT = np.mean(np.insert(onsets[1:],len(onsets[1:]),len(data))-peaks)/fs

	max_mag = np.max(data[peaks])
	min_mag = np.min(data[peaks])
	avg_mag = np.mean(data[peaks])
	
	max_amp = np.max(data[peaks]-data[onsets])
	min_amp = np.min(data[peaks]-data[onsets])
	avg_amp = np.mean(data[peaks]-data[onsets])
	f1 = [maxSCR,minSCR,avgSCR,freqSCR,durSCR]
	f2 = [avg_riseT,avg_recovT,max_mag,min_mag,avg_mag,max_amp,min_amp,avg_amp]
	f = f1+f2
	name = ['maSR','miSR','meSR','fqSR','durSR','meriseT', 'mecovT','mamag','mimag','memag','maamp','miamp','meamp']
	return f, name

def freq_feature(data,sig_type,fs):
	#===tonic===
	gsr_low_filter = low_pass_filter(data,fc=2,fs=fs, order=4)
	gsr_high_filter = high_pass_filter(gsr_low_filter,fc=0.01,fs=fs,order=4)
	f,pd = welch(gsr_high_filter, fs=fs, window='hamming', nperseg=1024,noverlap=64)
	if sig_type == 'p.':
		freq_band_name = ['VLF.','LF.','MF.','HF.','VHF.']
		freq_band = [[0.25,0.5],[0.5,0.75],[0.75,1],[1,1.25],[1.25,1.5]]
	elif sig_type == 't.':
		freq_band_name = ['LF.','HF.']
		freq_band = [[0,0.125],[0.125,0.25]]
	elif sig_type == 'o.':
		freq_band_name = ['VLF.','LF.','MF.','HF.','VHF.']
		freq_band = [[0.0,0.125],[0.125,0.25],[0.25,0.5],[0.5,0.75],[0.75,1]]
	feq_psd = []
	feq_psd_name = []
	feq_analysis_name = ['ro','s','ma','me','mi']
	all_psd = np.sum(pd[np.where(f<=freq_band[-1][-1])[0]])	
	for i, feq in enumerate(freq_band):
		feq_psd.append(np.sum(pd[band2idx(f,feq[0],feq[1])])/all_psd)
		feq_psd.append(np.sum(pd[band2idx(f,feq[0],feq[1])]))
		feq_psd.append(np.max(pd[band2idx(f,feq[0],feq[1])]))
		feq_psd.append(np.mean(pd[band2idx(f,feq[0],feq[1])]))
		feq_psd.append(np.min(pd[band2idx(f,feq[0],feq[1])]))
		feq_psd_name+=rename(feq_analysis_name,freq_band_name[i])
	#===phasic & tonic ===
	if sig_type == 'p.':
		phasic = band_pass_filter(data,lowcut=0.25, highcut=2, fs=fs, order=3)
		phasic = downsample(phasic,fs=16,nfs=4)
		f_p,pd_p = welch(phasic, fs=4, window='hamming', nperseg=1024)
		pd_p = pd_p[band2idx(f_p,0.25,2)]
		f_p = f_p[band2idx(f_p,0.25,2)]
		feature_type, feature_type_name = get_freq_info(pd_p,f_p)
	elif sig_type == 't.':
		tonic = band_pass_filter(data,lowcut=0.05, highcut=0.25, fs=fs, order=3)
		tonic = downsample(tonic,fs=16,nfs=1)
		f_t,pd_t = welch(tonic, fs=1, window='hamming', nperseg=1024)
		pd_t = pd_t[band2idx(f_t,0.05,0.25)]
		f_t = f_t[band2idx(f_t,0.05,0.25)]
		feature_type, feature_type_name = get_freq_info(pd_t,f_t)
	elif sig_type == 'o.':
		phasic = band_pass_filter(data,lowcut=0.25, highcut=2, fs=fs, order=3)
		phasic = downsample(phasic,fs=16,nfs=4)
		f_p,pd_p = welch(phasic, fs=4, window='hamming', nperseg=1024)
		pd_p = pd_p[band2idx(f_p,0.25,2)]
		f_p = f_p[band2idx(f_p,0.25,2)]
		feature_p, feature_p_name = get_freq_info(pd_p,f_p)
		tonic = band_pass_filter(data,lowcut=0.05, highcut=0.25, fs=fs, order=3)
		tonic = downsample(tonic,fs=16,nfs=1)
		f_t,pd_t = welch(tonic, fs=1, window='hamming', nperseg=1024)
		pd_t = pd_t[band2idx(f_t,0.05,0.25)]
		f_t = f_t[band2idx(f_t,0.05,0.25)]
		feature_t, feature_t_name = get_freq_info(pd_t,f_t)
		feature_type = feature_p + feature_t
		feature_type_name = rename(feature_p_name,'p.') + rename(feature_t_name,'t.')
	f = feq_psd+feature_type
	name = feq_psd_name+feature_type_name
	return f, name

def SCR_generate(signal,fs,min_amplitude,task):
	if task=='det.':
		onset, peak, amp, sig = detrendSCR(signal,fs,min_amplitude)	
	elif task == 'win.':
		onset, peak, amp, sig = windowSCR(signal,fs-1,min_amplitude)	
	elif task == 'df.':
		onset, peak, amp, sig = diffSCR(signal,fs,min_amplitude)
	elif task == 'bd.h.':
		onset, peak, amp, sig = freqSCR(signal,[0.5,2],'band',fs,min_amplitude)
	elif task == 'bd.m.':
		onset, peak, amp, sig = freqSCR(signal,[0.3,2],'band',fs,min_amplitude)
	elif task == 'bd.l.':
		onset, peak, amp, sig = freqSCR(signal,[0.1,2],'band',fs,min_amplitude)
	elif task == 'lo.h.':
		onset, peak, amp, sig = freqSCR(signal,[0.2],'low',fs,min_amplitude)		
	elif task == 'lo.l.':
		onset, peak, amp, sig = freqSCR(signal,[0.08],'low',fs,min_amplitude)
	elif task == 'CDA.':
		onset, peak, amp, sig, sig2 = CDASCR(signal,fs,0.3)
		return onset, peak, amp, sig, sig2	
	elif task == 'CVX.':
		onset, peak, amp, sig, sig2 = CVXSCR(signal,fs,min_amplitude)
		return onset, peak, amp, sig, sig2	
	return onset, peak, amp, sig

def entropy_feature(data,match,scale):
	std = np.std(data)
	ap_ent = ap_entropy(data,match=match,tolerance=0.2*std)
	inf_ent = information_entropy(data,match=match)
	rcmse_ent = []
	name1 = []
	rcmpe_ent = []
	name2 = []
	for i in range(1,scale+1):
		rcmse_ent.append(RCMSE(data,match=match,scale=i,tolerance=0.2*std))
		name1.append('rms.'+str(i))
		rcmpe_ent.append(RCMPE(data,match=match,scale=i,delay=1))
		name2.append('rmp.'+str(i))
	f = [ap_ent, inf_ent]+rcmse_ent+rcmpe_ent
	name = ['ApEn', 'InEn']+name1+name2
	return f, name

def DWT(data):
	sig = pywt.wavedec(data, 'db3', mode='symmetric', level=5, axis=-1)
	f = []
	name = []
	freq = ['.H','.M','.L']
	for i in range(3):
		f.append(np.mean(abs(sig[i])))
		f.append(np.std(sig[i]))
		f.append(np.sum(sig[i]**2)/len(sig[i]))
		f.append(skew(sig[i]))
		f.append(kurtosis(sig[i]))
		f.append(ap_entropy(sig[i],match=2,tolerance=0.2*np.std(sig[i])))
		f.append(information_entropy(sig[i],match=2))
		name.append('DWT.me.E'+freq[i])
		name.append('DWT.std'+freq[i])
		name.append('DWT.me.P'+freq[i])
		name.append('DWT.sk'+freq[i])
		name.append('DWT.ku'+freq[i])
		name.append('DWT.ApEn'+freq[i])
		name.append('DWT.InEn'+freq[i])
	return f, name
	



