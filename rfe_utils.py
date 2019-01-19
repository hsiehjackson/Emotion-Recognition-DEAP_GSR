import numpy as np 
import os 
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, log_loss
from sklearn.svm import SVC,LinearSVC
from sklearn.feature_selection import RFE

collectnum = 1

def getbestfeature(task,a_feature,v_feature):
	bestfeature = []
	vote = []
	if task == 'arousal':
		for j in sorted(a_feature, key=lambda i: len(a_feature[i]), reverse=True):
			bestfeature.append(j)
			vote.append(len(a_feature[j]))
			if len(bestfeature)==20:
				break
		print('arousal feature: ',bestfeature)
	
	elif task =='valence':
		for j in sorted(v_feature, key=lambda i: len(v_feature[i]), reverse=True):
			bestfeature.append(j)
			vote.append(len(v_feature[j]))
			if len(bestfeature)==20:
				break
		print('valence feature: ',bestfeature)
	return vote, bestfeature

def allfeature_importance(subject,sort_importance,importance,task, feature_name,a_feature,v_feature):
	if task=='arousal':
		for i in range(collectnum):
			find = np.where(importance==sort_importance[i])[0]
			for j in find:
				name = feature_name[j]
				if name not in a_feature:
					a_feature[name]=[subject]
				else:
					a_feature[name].append(subject)
	
	elif task == 'valence':
		for i in range(collectnum):
			find = np.where(importance==sort_importance[i])[0]
			for j in find:
				name = feature_name[j]
				if name not in v_feature:
					v_feature[name]=[subject]
				else:
					v_feature[name].append(subject)	
	return a_feature, v_feature

def train(X_train, Y_train, X_val, Y_val, subject, threshold, task, feature_name, method, a_feature,v_feature):

	#print('traing...')
	train_acc = np.zeros(threshold)
	train_f1 = np.zeros(threshold)
	val_acc = np.zeros(threshold)
	val_f1 = np.zeros(threshold)
	val_roc = np.zeros(threshold)
	feature_size = np.zeros(threshold)

	xgb1result, train_acc[0], train_f1[0], val_acc[0], val_f1[0], val_roc[0] = modelfit(X_train, Y_train, X_val, Y_val, len(X_train[0]))
	ranking = xgb1result.ranking_
	sort_ranking = np.sort(np.unique(importance))
	a_feature, v_feature = allfeature_importance(subject, sort_ranking, ranking, task, feature_name, a_feature, v_feature)


	for thr in range(5,threshold+1):
		if thr==0:
			print(task)
			print("feat_size: %.3f train f1: %.3f val f1: %.3f roc auc: %.3f" % (feature_size[thr],train_f1[thr], val_f1[thr], val_roc[thr]))
		else:
			xgb2result, train_acc[thr], train_f1[thr], val_acc[thr], val_f1[thr], val_roc[thr] = modelfit(X_train, Y_train, X_val, Y_val,thr)


			print("feat_size: %.3f train f1: %.3f val f1: %.3f roc auc: %.3f" % (feature_size[thr],train_f1[thr], val_f1[thr], val_roc[thr]))
	

	return train_acc, train_f1, val_acc, val_f1, val_roc,feature_size, a_feature, v_feature

def modelfit(X_train, Y_train, X_val, Y_val, H):

	
	#alg = SVC(C=0.01, kernel='rbf',gamma='auto',probability=True)
	model = SVC(C=0.01, kernel='linear',probability=True)
	alg = RFE(model,H,step=1)

	alg.fit(X_train, Y_train)
	Ytrain_pred = alg.predict(X_train)
	Ytrain_pred = (Ytrain_pred>0.5)*1
	Yval_pred = alg.predict(X_val)
	Yval_pred = (Yval_pred>0.5)*1
	Ytrain_pred_prob  = alg.predict_proba(X_train)
	Yval_pred_prob = alg.predict_proba(X_val)

	train_acc = accuracy_score(Y_train,Ytrain_pred) 
	train_f1 = f1_score(Y_train,Ytrain_pred,average='binary')
	#train_loss = log_loss(Y_train,Ytrain_pred_prob)
	val_acc = accuracy_score(Y_val,Yval_pred)
	val_f1 = f1_score(Y_val,Yval_pred,average='binary')
	val_roc = roc_auc_score(Y_val,Yval_pred_prob[:,1])
	#val_loss = log_loss(Y_val, Yval_pred_prob)

	return alg, train_acc, train_f1, val_acc, val_f1, val_roc