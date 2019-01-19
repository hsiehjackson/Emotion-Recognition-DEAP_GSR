import numpy as np 
import os 
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, log_loss
from sklearn.svm import SVC,LinearSVC


def train(X_train, Y_train, X_val, Y_val, subject, threshold, task, feature_name, method,a_feature,v_feature):

	#print('traing...')
	train_acc = np.zeros(threshold)
	train_f1 = np.zeros(threshold)
	val_acc = np.zeros(threshold)
	val_f1 = np.zeros(threshold)
	val_roc = np.zeros(threshold)
	feature_size = np.zeros(threshold)

	xgb1result, train_acc[0], train_f1[0], val_acc[0], val_f1[0], val_roc[0] = modelfit(X_train, Y_train, X_val, Y_val)

	for thr in range(threshold):
		print(task)
		print("feat_size: %.3f train f1: %.3f val f1: %.3f roc auc: %.3f" % (feature_size[thr],train_f1[thr], val_f1[thr], val_roc[thr]))
		#print("train acc: %.3g, train f1: %.3f" %(train_acc[thr], train_f1[thr]))
		#print("val acc: %.3g, val f1: %.3f" %(val_acc[thr], val_f1[thr]))
		#print("roc auc: %.3g" % val_roc[thr])

	return train_acc, train_f1, val_acc, val_f1, val_roc,feature_size, a_feature, v_feature

def modelfit(X_train, Y_train, X_val, Y_val):

	alg = SVC(C=0.01, kernel='linear',probability=True)
	#alg = SVC(C=0.01, kernel='rbf',gamma='auto',probability=True)
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