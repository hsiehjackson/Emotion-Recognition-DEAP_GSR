import sys 
import os
import csv
import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
import _pickle as pk

folder = '/home/edan/Jackson/Deap_dataset/data_origin_txt/'
savefolder = '/home/edan/Jackson/Deap_dataset/data_preprocessed_pickle/'
labelfile = '/home/edan/Jackson/Deap_dataset/zip/participant_ratings.csv'

def downsample(data, fs, nfs):
	secs = float(len(data)/fs)
	sample = int(secs*nfs)
	newdata =  resample(data, sample)
	return newdata


def readsignal(original):
	print('read signal...')
	signal = []
	file = open(original, 'r')
	file.readline()
	for line in csv.reader(file):
		signal.append(line[41])
	file.close()
	signal = np.array(signal).astype('float')
	signal = downsample(signal,512,128)
	print('all signal: ', signal.shape)
	return signal

def readannotation(annotations):
	print('read annotation...')
	file = open(annotations,'r')
	file.readline()
	all_annotations = [] 
	exp_annotations = []
	for line in csv.reader(file):
		all_annotations.append([float(line[0]),int(line[1][-1])])
	file.close()
	for n, item in enumerate(all_annotations[3:]):
		if all_annotations[n-3][1]==2 and all_annotations[n-2][1]==3 and all_annotations[n-1][1]==1 and all_annotations[n][1]==3:
			exp_annotations.append([all_annotations[n-3][0],all_annotations[n][0]])
			if int(all_annotations[n][0] - all_annotations[n-3][0])!=65:
				print('======',int(all_annotations[n][0] - all_annotations[n-3][0]),'======')
	exp_annotations = np.array(exp_annotations)
	print('annotation: ', exp_annotations.shape)
	return exp_annotations


def readlabel(label):
	order_exp = []
	all_label = []
	file = open(label,'r')
	file.readline()
	for line in csv.reader(file):
		order_exp.append(line[2])
		all_label.append(line[4:8])
	order_exp = np.array(order_exp).astype('int')
	all_label = np.array(all_label).astype('float')
	return order_exp, all_label

def getsignal(signal, annotations, order_exp, label):
	sig_all = []
	for n, time in enumerate(annotations):
		sig = signal[int(time[0]*128):int(time[0]*128)+65*128]
		sig_all.append(sig)
	sig_all = np.array(sig_all)
	print('exp signal:', sig_all.shape)
	order_sig_all = np.zeros_like(sig_all).astype('float')
	order_label_all = np.zeros_like(label).astype('float')
	for n, exp in enumerate(order_exp):
		order_sig_all[exp-1] = sig_all[n]
		order_label_all[exp-1] = label[n]
	sig_dict = {}
	sig_dict['data'] = order_sig_all
	sig_dict['label'] = order_label_all
	return sig_dict

def main():
	order_exp, label = readlabel(labelfile)
	print('order exp: ',order_exp.shape)
	print('label: ', label.shape)
	for n in range(16,23):
		name = 's'+str(n).rjust(2,'0')
		print(name)
		signal = readsignal(folder+name+'_data.txt')
		annotations =  readannotation(folder+name+'_annotations.txt')
		sig_dict = getsignal(signal, annotations, order_exp[(n-1)*40:(n)*40], label[(n-1)*40:(n)*40])
		
		print('save... {} data:{} label:{}'.format(name,sig_dict['data'].shape,sig_dict['label'].shape))
		file = open(savefolder+name+'.pkl','wb')
		pk.dump(sig_dict,file)
		file.close()
			

if __name__ == '__main__':
	main()