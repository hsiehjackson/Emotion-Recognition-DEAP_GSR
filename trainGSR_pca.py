import csv
import random
from sys import argv
import warnings  
from xgb_utils2 import *
import scipy.stats as stats
from multiprocessing import set_start_method
from sklearn.decomposition import PCA
"======setting======" 
threshold = 3 #different feature number
fold = 22
# participant (1-22)
# exp (1-40)
# 0 exp_number
# 1-207 feature !!!!we don't want first feature
# 416 arousal, 415 valence
"======setting======"

def p_value(x,y):
	l_0 = []
	l_1 = []
	for i in range(len(y)):
		if(y[i]==0):
			l_0.append(x[i])
		else:
			l_1.append(x[i])
	f, p1 = stats.f_oneway(l_0, l_1)
	f, p2 = stats.ttest_ind(l_0, l_1)       
	return f, p1

def calculateP(name, label_all, data_all,task):
	print('write '+task+' P value...')
	mean = np.mean(data_all,axis=0)
	std = np.std(data_all,axis=0)
	data_all = (data_all-mean)/std
	p1 = []
	p2 = []
	for i in range(data_all.shape[1]):
		p_1,p_2 = p_value(data_all[:,i],label_all)
		p1.append(p_1)
		p2.append(p_2)
	file = open('gsr_result/gsr_fusion/gsr_pvalue/'+task+'_pv.csv','w')
	for i in range(len(p1)):
		file.write(str(i)+','+str(name[i])+','+str(p1[i])+','+str(p2[i]))
		file.write('\n')
	file.close()

def onehot(label):
	newlabel = np.zeros(len(label)).astype('int')
	allmean = 0
	for i in range(len(label)):
		j= i//40
		mean = np.mean(label[j*40:j*40+40])
		allmean+=mean
		newlabel[i] = int((label[i]>mean)*1)
	allmean = allmean/len(label)
	return allmean, newlabel

def data_shuffle(X, Y):
	pair = list(zip(X,Y))
	random.shuffle(pair)
	X, Y = zip(*pair)
	return np.array(X), np.array(Y)

def feature_clean(data, name):
	for column in range(data.shape[1],0,-1):
		for n, element in enumerate(data[:,column-1]):
			if str(element) == 'nan':
				data = np.delete(data,column-1,1)
				name = np.delete(name,column-1)
				break
	data = data.astype('float')
	for column in range(data.shape[1],0,-1):
		for participant in range(len(data)//40):
			if np.std(data[participant*40:(participant+1)*40,column-1])==0:
				data = np.delete(data,column-1,1)
				name = np.delete(name,column-1)
				break
	return data, name

def readfile(filename, task):
	file = open(filename,'r')
	a_label_all = []
	v_label_all = []
	data_all = []
	alllist = list(csv.reader(file))
	for n, row in enumerate(alllist[1:]):
		v_label_all.append(float(row[2695]))
		a_label_all.append(float(row[2696]))
		data_all.append(row[1:2695])		
	a_label_all = np.array(a_label_all)
	v_label_all = np.array(v_label_all)
	data_all = np.array(data_all)
	file.close()
	name = np.array(alllist[0][1:2695])
	data_all, name = feature_clean(data_all[:,task[0]:task[1]], name[task[0]:task[1]])

	for i in range(len(data_all)//40):
		mean = np.mean(data_all[i*40:(i+1)*40],axis=0)
		std = np.std(data_all[i*40:(i+1)*40],axis=0)
		data_all[i*40:(i+1)*40] = (data_all[i*40:(i+1)*40] - mean)/std

	return name, a_label_all, v_label_all ,data_all

def writefile(filename, accuracy,task,write):
	writefile = open(filename,write)
	writefile.write(task+'\n')
	writefile.write('feature_size, train_acc, train_f1, val_acc, val_f1, val_roc\n')
	for i in range(len(accuracy)):
		writefile.write(','.join(repr(accuracy[i][j]) for j in range(len(accuracy[i]))))
		writefile.write('\n')
	writefile.close()

def savePC(PC,filename):
	file = open(filename,'w')
	file.write(str(len(PC))+'\n')
	for p_c in PC:
		file.write(str(p_c)+'\n')
	file.close()

def writefeat(filename,feat,vote,task,write):
	writefile = open(filename,write)
	writefile.write(task+'\n')
	writefile.write(','.join(str(feat[i]) for i in range(len(feat))))
	writefile.write('\n')
	writefile.write(','.join(str(vote[i]) for i in range(len(vote))))
	writefile.write('\n')
	writefile.close()

def main():
	'''
	task = [0, 134, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
	out = ['ori','det','win','df','bd.h','bd.m','bd.l','lo.h','lo.l','CDA','CVX']
	out = [x +'_pca2.csv' for x in out]
	for i in range(1,len(task)):
		sumdata = sum(task[0:i])
		trainTask([sumdata,sumdata+task[i]],out[i-1])
	'''
	trainTask([0,2694],'fusion_pcath.csv')

def trainTask(task,outfile):
	print(outfile,'...')
	a_feature = {}
	v_feature = {}
	name, a_label_all, v_label_all, data_all = readfile(infile,task)
	#print('data: {} alabel {} vlable {}'.format(data_all.shape, a_label_all.shape, v_label_all.shape))
	#input()
	"======p_value=====onehot======"
	a_cut, a_label_all = onehot(a_label_all)
	v_cut, v_label_all = onehot(v_label_all)
	#calculateP(name, a_label_all, data_all,'a.'+outfile[:-4])
	#calculateP(name, v_label_all, data_all,'v.'+outfile[:-4])
	#input()
	"===========variable==========="
	a_train_acc = np.zeros((threshold,fold))
	a_train_f1 = np.zeros((threshold,fold))
	a_val_f1 = np.zeros((threshold,fold))
	a_val_acc = np.zeros((threshold,fold))
	a_val_roc = np.zeros((threshold,fold))
	a_feature_size = np.zeros((threshold,fold))
	v_train_acc = np.zeros((threshold,fold))
	v_train_f1 = np.zeros((threshold,fold))
	v_val_f1 = np.zeros((threshold,fold))
	v_val_acc = np.zeros((threshold,fold))
	v_val_roc = np.zeros((threshold,fold))
	v_feature_size = np.zeros((threshold,fold))
	"======35 fold validation======"
	folder = 'gsr_result/gsr_fusion/report/img/pca/'+outfile[:-4]+'/'
	if not os.path.exists(folder):
		os.mkdir(folder)
	print('PCA...')
	mean = np.mean(data_all,axis=0)
	std = np.std(data_all,axis=0)
	data_all = (data_all-mean)/std
	pca = PCA(n_components=0.99,svd_solver='full')
	data_all = pca.fit_transform(data_all)
	PC = pca.fit(data_all)
	cum = np.cumsum(PC.explained_variance_ratio_)
	savePC(cum,folder+'pca.csv')
	plt.plot(cum)
	plt.savefig(folder+'pca.png')
	plt.close()
	
	for val in list(range(fold)):
		#print('============================')
		print('val subject ', val)
		"===========train and val==========="
		train_index = list(range(0, val*40)) + list(range(val*40+40, len(data_all)))
		val_index = list(range(val*40, val*40+40))
		X_train = data_all[train_index]
		X_val = data_all[val_index]
		a_Y_train = a_label_all[train_index]
		a_Y_val = a_label_all[val_index]
		v_Y_train = v_label_all[train_index]
		v_Y_val = v_label_all[val_index]
		"===========normalize==========="
		mean = np.mean(X_train,axis=0)
		std = np.std(X_train,axis=0)
		X_train = (X_train-mean)/std
		X_val = (X_val-mean)/std
		"==========shuffle=========="
		a_X_train = X_train
		v_X_train = X_train
		a_X_train, a_Y_train = data_shuffle(a_X_train,a_Y_train)
		v_X_train, v_Y_train = data_shuffle(v_X_train,v_Y_train)
		"===========train==========="
		a_tn_acc, a_tn_f1, a_v_acc, a_v_f1, a_roc, a_feat_size,a_feature,v_feature = train(a_X_train,a_Y_train,X_val,a_Y_val, val, threshold,'arousal',name,outfile[:-4],a_feature,v_feature)
		a_train_acc[:,val] = a_tn_acc
		a_train_f1[:,val] = a_tn_f1
		a_val_acc[:,val] = a_v_acc
		a_val_f1[:,val] = a_v_f1
		a_val_roc[:,val] = a_roc
		a_feature_size[:,val] = a_feat_size
		v_tn_acc, v_tn_f1, v_v_acc, v_v_f1, v_roc, v_feat_size,a_feature,v_feature = train(v_X_train,v_Y_train,X_val,v_Y_val, val, threshold,'valence',name,outfile[:-4],a_feature,v_feature)
		v_train_acc[:,val] = v_tn_acc
		v_train_f1[:,val] = v_tn_f1
		v_val_acc[:,val] = v_v_acc
		v_val_f1[:,val] = v_v_f1
		v_val_roc[:,val] = v_roc
		v_feature_size[:,val] = v_feat_size
		#cutline.append(cut)
		#one.append(onetrain+oneval)
		#zero.append(zerotrain+zeroval)

	"===========getresult==========="
	#allonezero = sum(one)+sum(zero)	
	#one = sum(one)/allonezero
	#zero = sum(zero)/allonezero
	#cutline = np.mean(np.array(cutline))
	a_train_acc = np.sum(a_train_acc,axis=1)/(fold)
	a_train_f1 = np.sum(a_train_f1,axis=1)/(fold)
	a_val_acc = np.sum(a_val_acc,axis=1)/(fold)
	a_val_f1 = np.sum(a_val_f1,axis=1)/(fold)
	a_val_roc = np.sum(a_val_roc,axis=1)/(fold)
	a_feature_size = np.sum(a_feature_size,axis=1)/(fold)

	v_train_acc = np.sum(v_train_acc,axis=1)/(fold)
	v_train_f1 = np.sum(v_train_f1,axis=1)/(fold)
	v_val_acc = np.sum(v_val_acc,axis=1)/(fold)
	v_val_f1 = np.sum(v_val_f1,axis=1)/(fold)
	v_val_roc = np.sum(v_val_roc,axis=1)/(fold)
	v_feature_size = np.sum(v_feature_size,axis=1)/(fold)
	
	a_one = len(np.where(a_label_all==int(1))[0])/len(a_label_all)
	v_one = len(np.where(v_label_all==int(1))[0])/len(v_label_all)
	print("===============================")
	print('arousal:') 
	print('one: %2f zero: %2f cutline: %2f' % (a_one,1-a_one,a_cut))
	print('feature_size = '+str(a_feature_size))
	print('train accuracy = '+str(a_train_acc))
	print('train f1 score = '+str(a_train_f1))
	print('test accuracy = '+str(a_val_acc))
	print('test f1 score = '+str(a_val_f1))
	print('test roc score = '+str(a_val_roc))
	#vote, a_best_feature = getbestfeature('arousal',a_feature,v_feature)

	print("===============================")
	print('valence:')
	print('one: %2f zero: %2f cutline: %2f' % (v_one,1-v_one,v_cut))
	print('feature_size = '+str(v_feature_size))
	print('train accuracy = '+str(v_train_acc))
	print('train f1 score = '+str(v_train_f1))
	print('test accuracy = '+str(v_val_acc))
	print('test f1 score = '+str(v_val_f1))
	print('test roc score = '+str(v_val_roc))
	#vote, v_best_feature = getbestfeature('valence',a_feature,v_feature)

	a_accuracy = np.array([a_feature_size, a_train_acc, a_train_f1, a_val_acc, a_val_f1, a_val_roc]).T
	v_accuracy = np.array([v_feature_size, v_train_acc, v_train_f1, v_val_acc, v_val_f1, v_val_roc]).T
	writefile('gsr_result/gsr_fusion/report/'+outfile,a_accuracy,'arousal','w')
	writefile('gsr_result/gsr_fusion/report/'+outfile,v_accuracy,'valence','a')
	#writefeat('gsr_result/gsr_fusion/report/ft_'+outfile,a_best_feature,vote,'arousal','w')
	#writefeat('gsr_result/gsr_fusion/report/ft_'+outfile,v_best_feature,vote,'valence','a')
	
if __name__ == '__main__':
	warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)  
	infile = argv[1]
	#outfile = argv[2]
	main()

