import numpy as np
import time, math
import random
import torch
from torch.utils.data import TensorDataset,DataLoader

def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def split_data(X, Y, ratio):
	pair = list(zip(X,Y))
	random.shuffle(pair)
	X, Y = zip(*pair)
	X = np.array(X)
	Y = np.array(Y)
	val_size = int(len(X) * ratio)
	return X[val_size:],Y[val_size:],X[:val_size],Y[:val_size]

def getloader(data, label, batch_size):
	data = torch.from_numpy(data.astype('float')).type(torch.FloatTensor)
	label = torch.from_numpy(label.astype('int')).type(torch.LongTensor)
	dataset = TensorDataset(data,label)
	loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
	return loader

def onehot(label):
	newlabel = np.zeros(len(label)).astype('int')
	allmean = 0
	for i in range(len(label)):
		j= i//40
		mean = np.mean(label[j*40:j*40+40])
		allmean+=mean
		newlabel[i] = int((label[i]>mean)*1)
	allmean = allmean/len(label)
	return newlabel