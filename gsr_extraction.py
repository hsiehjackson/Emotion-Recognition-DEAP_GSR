import os,csv
from gsr_utils import *
from gsr_feature import *
import warnings
from sys import argv
import _pickle as pk

#[134, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256] 2694
PARAPLOT = False
datafolder_bad = 'data_preprocessed_python/'
datafolder_good = 'data_preprocessed_pickle/'
writefile = argv[1]

def readfile(filename, quality):
	signal = pk.load(open(filename, 'rb'),encoding='latin1')
	data = signal['data'] 
	if quality=='bad':
		labels = signal['labels']
		gsr_signal = data[:,36,:]
	elif quality == 'good':
		labels = signal['label']
		gsr_signal = data
	return gsr_signal, labels

def writesignal(filename, participant ,signal):
	file = open(filename, 'a', encoding = 'big5')
	file.write(participant)
	for i in range(len(signal)):
		file.write(','+str(signal[i]))
	file.write('\n')
	file.close()

def feature_set(data,sig_type,fs):
	print('    get statistics_feature...')
	f1, name1 = statistics_feature(data)
	f2, name2 = statistics_feature(np.diff(data))
	f3, name3 = statistics_feature(np.diff(np.diff(data)))
	name2 = rename(name2, '1df.')
	name3= rename(name3, '2df.')
	print('    get freq_feature...')
	f4, name4 = freq_feature(data,sig_type,fs)
	print('    get DWT_feature...')
	f5, name5 = DWT(data)
	print('    get entropy_feature...')
	f6, name6 = entropy_feature(data,match=2,scale=20)
	f = f1+f2+f3+f4+f5+f6
	name = name1+name2+name3+name4+name5+name6 
	name = rename(name, sig_type)
	return f, name

def main():
	labelname = ['Valence','Arousal','Dominance','Liking']
	writename = True
	for participant in range(1,23):
		filename_bad = 's%02d.dat' % (participant)
		filename_good = 's%02d.pkl' % (participant)
		gsr_signal_bad, labels_bad = readfile(datafolder_bad + filename_bad, 'bad')
		gsr_signal_good, labels_good = readfile(datafolder_good + filename_good, 'good')
		for exp in range(1,2):
			if PARAPLOT == True:
				plt.subplot(121)
				plt.plot(gsr_signal_bad[exp,:])
				plt.subplot(122)
				plt.plot(gsr_signal_good[exp,:])
				plt.show()	
			
			filename = str(participant)+'_'+str(exp+1)
			all_feature = []
			all_name = []
			print(filename)
			gsr_us_conductance = gsr_signal_good[exp,:]
			gsr_filter = low_pass_filter(gsr_us_conductance, fc=2, fs=128, order=5)
			SC = downsample(gsr_filter,fs=128,nfs=16)
			SC = SC/1e3
			file = open('test.csv','w')
			for i in gsr_filter/1e3:
				file.write(str(i)+'\n')
			file.close()
			df = np.diff(SC)
			size = int(1. * 16)
			df = tools.smoother(df,'bartlett',size)['signal']
			zeros, = tools.zero_cross(signal=df, detrend=False)
			if len(zeros)==0:
				gsr_filter = detrend(gsr_filter)+np.min(gsr_filter)
				SC = detrend(SC)+np.min(SC)

			feature_size = []

			#for task in ['ori.','det.','win.','df.','bd.h.','bd.m.','bd.l.','lo.h.','lo.l.','CDA.','CVX.']:
			for task in ['CDA.']:
				print('get '+task+'feature...')
				if task == 'ori.':
					feature, name = feature_set(gsr_filter/1e3, 'o.', fs=128)
				elif task == 'CDA.':
					onset, peak, amp, driver, phasic = SCR_generate(gsr_filter/1e3,fs=128,min_amplitude=0.1,task=task)	
					'''
					tonic = SC - phasic
					#plotpeak(SC,driver,tonic,onset,peak,amp,filename)
					f1, name1 = SCR_feature(driver,onset,peak,amp,fs=16)
					print('  get phasic...')
					f2, name2 = feature_set(phasic,'p.',fs=16)
					print('  get tonic...')
					f3, name3 = feature_set(tonic,'t.',fs=16)
					feature = f1+f2+f3
					name = name1+name2+name3
					name = rename(name,task)
				elif task == 'CVX.':
					onset, peak, amp, phasic, tonic = SCR_generate(SC,fs=16,min_amplitude=0.1,task=task)	
					plotpeak(SC,phasic,tonic,onset,peak,amp,filename)
					f1, name1 = SCR_feature(phasic,onset,peak,amp,fs=16)
					print('  get phasic...')
					f2, name2 = feature_set(phasic, 'p.',fs=16)
					print('  get tonic...')
					f3, name3 = feature_set(tonic,'t.',fs=16)
					feature = f1+f2+f3
					name = name1+name2+name3
					name = rename(name,task)
				else:
					onset, peak, amp, phasic = SCR_generate(SC,fs=16,min_amplitude=0.1,task=task)
					tonic = SC - phasic
					#plotpeak(SC,phasic,tonic,onset,peak,amp,filename)	
					f1, name1 = SCR_feature(phasic,onset,peak,amp,fs=16)
					print('  get phasic...')
					f2, name2 = feature_set(phasic, 'p.',fs=16)
					print('  get tonic...')
					f3, name3 = feature_set(tonic,'t.',fs=16)
					feature = f1+f2+f3
					name = name1+name2+name3
					name = rename(name,task)
				if len(feature_size)<11:
					feature_size.append(len(feature))
				all_feature += feature
				all_name += name				
			print(feature_size,sum(feature_size))
			all_feature += labels_good[exp,:].tolist()
			all_name += labelname
			print(len(all_feature))
			if writename == False:
				writesignal(writefile,'',all_name)	
				file = open('feature_all.csv', 'w')
				for i,n in enumerate(all_name):
					file.write(str(n)+'\n')
				file.close()
				writename = True	
			writesignal(writefile,filename ,all_feature)
			print('==========')
			'''
if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	main()


