import _pickle as pk
from sys import argv
import os
import numpy as np
from torch_util import *
from torch_model import *
from torch_path import *
from torch_trainer import fit
from torch.optim import lr_scheduler

use_cuda = torch.cuda.is_available()

Val_Ratio = 0.1
BATCH_SIZE = 8
EPOCH = 50
Learning_Rate = 1e-4
File_Folder = argv[1]

def readfile(filename):
	signal = pk.load(open(filename, 'rb'),encoding='latin1')
	data = signal['data'] 
	labels = signal['label']
	return data, labels

def main():
	signal = pk.load(open(gsrfile,'rb'))
	data = signal['data']
	label = signal['label']
	#normalize
	mean = np.mean(data,axis=1)
	std = np.std(data,axis=1)
	data = ((data.T-mean)/std.T).T
	valence = onehot(label[:,0])
	arousal = onehot(label[:,1])
	#getloader
	X_train, Y_train, X_val, Y_val = split_data(data,arousal,ratio=Val_Ratio)
	train_loader = getloader(X_train, Y_train, BATCH_SIZE)
	val_loader = getloader(X_val, Y_val, BATCH_SIZE)

	#train
	device = torch.device("cuda" if use_cuda else "cpu")
	if not os.path.exists(ckpt_path+File_Folder+'/'):
		os.mkdir(ckpt_path+File_Folder)
	loss_fn = nn.CrossEntropyLoss()
	model = CNN_Classifier().to(device)
	model = Balence_Net_Classifier().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate, betas=(0.9, 0.999))
	scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
	fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, EPOCH, use_cuda, File_Folder)

if __name__ == '__main__':
	main()
