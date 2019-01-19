import numpy as np 
import xgboost as xgb
import os 
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, log_loss
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pandas as pd


collectnum = 2

def getbestfeature(task,a_feature,v_feature):
	bestfeature = []
	vote = []
	if task == 'arousal':
		for j in sorted(a_feature, key=lambda i: len(a_feature[i]), reverse=True):
			bestfeature.append(j)
			vote.append(len(a_feature[j]))
			if len(bestfeature)==10:
				break
		print('arousal feature: ',bestfeature)
	
	elif task =='valence':
		for j in sorted(v_feature, key=lambda i: len(v_feature[i]), reverse=True):
			bestfeature.append(j)
			vote.append(len(v_feature[j]))
			if len(bestfeature)==10:
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

def train(X_train, Y_train, X_val, Y_val, subject, threshold, task, feature_name, method,a_feature,v_feature):

	#print('traing...')
	train_acc = np.zeros(threshold)
	train_f1 = np.zeros(threshold)
	val_acc = np.zeros(threshold)
	val_f1 = np.zeros(threshold)
	val_roc = np.zeros(threshold)
	feature_size = np.zeros(threshold)

	xgb1result, train_acc[0], train_f1[0], val_acc[0], val_f1[0], val_roc[0] = modelfit(X_train, Y_train, X_val, Y_val)
	importance = (xgb1result.feature_importances_)
	importance = np.sort(np.unique(importance))[::-1]
	#print('Ploting...')
	"==============plot=============="
	folder = 'gsr_result/gsr_fusion/report/img/feature_importance/'+method+'/'
	if not os.path.exists(folder):
		os.mkdir(folder)
	'''
	fig, ax = plt.subplots(1,1)
	mapper = { i_d : name for i_d, name in enumerate(feature_name)}
	mapped = {mapper[i_d]: value for i_d, value in enumerate(xgb1result.feature_importances_)}
	feat_imp = pd.Series(mapped).sort_values(ascending=False)
	feat_imp = feat_imp[:30]
	plt.rc('xtick', labelsize=5)
	feat_imp.plot(kind='bar', title='Feature Importances', color='b')
	plt.ylabel('Feature Importance Score')
	plt.savefig(folder+task+str(subject)+'.png')
	plt.close()

	fig, ax = plt.subplots(1,1)
	xgb.plot_importance(xgb1result, max_num_features=10, ax=ax)
	fig.savefig(folder+task+str(subject)+'.png')
	plt.close()
	'''	
	for thr in range(threshold):
		if thr==0:
			feature_size[thr] = X_train.shape[1]
			#print("Thresh=%.6f, n=%d" %(thr, X_train.shape[1]))
			print(task)
			print("feat_size: %.3f train f1: %.3f val f1: %.3f roc auc: %.3f" % (feature_size[thr],train_f1[thr], val_f1[thr], val_roc[thr]))
			#print("train acc: %.3g, train f1: %.3f" %(train_acc[thr], train_f1[thr]))
			#print("val acc: %.3g, val f1: %.3f" %(val_acc[thr], val_f1[thr]))
			#print("roc auc: %.3g" % val_roc[thr])
		else:
			selection = SelectFromModel(xgb1result, threshold=importance[thr-1], prefit=True)
			select_X_train = selection.transform(X_train)
			select_X_val = selection.transform(X_val)
			feature_size[thr] = select_X_train.shape[1]
			xgb2result, train_acc[thr], train_f1[thr], val_acc[thr], val_f1[thr], val_roc[thr] = modelfit(select_X_train, Y_train, select_X_val, Y_val)
			#print("Thresh=%.6f, n=%d" %(thr, select_X_train.shape[1]))
			print("feat_size: %.3f train f1: %.3f val f1: %.3f roc auc: %.3f" % (feature_size[thr],train_f1[thr], val_f1[thr], val_roc[thr]))
			#print("train acc: %.3g, train f1: %.3f" %(train_acc[thr], train_f1[thr]))
			#print("val acc: %.3g, val f1: %.3f" %(val_acc[thr], val_f1[thr]))
			#print("roc auc: %.3g" % val_roc[thr])
	
	#a_feature, v_feature = allfeature_importance(subject, importance, xgb1result.feature_importances_, task, feature_name,a_feature,v_feature)
	return train_acc, train_f1, val_acc, val_f1, val_roc,feature_size, a_feature, v_feature


def modelfit(X_train, Y_train, X_val, Y_val, cv_folds=10, early_stopping_rounds=50):
	metrics = 'logloss'
	#metrics = 'auc'

	alg = getbestparameters(X_train,Y_train,plot=False)
	alg.set_params(learning_rate=0.01)
	xgb_param = alg.get_xgb_params()
	dtrain = xgb.DMatrix(X_train, label=Y_train)
	cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=5000, nfold=cv_folds,
	metrics=metrics, early_stopping_rounds=early_stopping_rounds,verbose_eval=False,shuffle=True)
	alg.set_params(n_estimators=cvresult.shape[0])
	alg.fit(X_train, Y_train, eval_metric=metrics)
	#xgb.plot_tree(alg)
	#plt.show()
	Ytrain_pred = alg.predict(X_train)
	Ytrain_pred = (Ytrain_pred>0.5)*1
	Yval_pred = alg.predict(X_val)
	Yval_pred = (Yval_pred>0.5)*1
	Ytrain_pred_prob  = alg.predict_proba(X_train)
	Yval_pred_prob = alg.predict_proba(X_val)

	train_acc = accuracy_score(Y_train,Ytrain_pred) 
	train_f1 = f1_score(Y_train,Ytrain_pred,average='binary')
	train_loss = log_loss(Y_train,Ytrain_pred_prob)
	val_acc = accuracy_score(Y_val,Yval_pred)
	val_f1 = f1_score(Y_val,Yval_pred,average='binary')
	val_roc = roc_auc_score(Y_val,Yval_pred_prob[:,1])
	val_loss = log_loss(Y_val, Yval_pred_prob)
	#print('train loss: {} val loss: {}'.format(train_loss,val_loss))
	#print('===============')
	#print('True: ',Y_val)
	#print('Pred: ',Yval_pred)
	return alg, train_acc, train_f1, val_acc, val_f1, val_roc

def getbestparameters(X_train, Y_train, plot):
	metrics = 'logloss' 
	#metrics = 'auc'
	scoring = 'neg_log_loss'
	#scoring = None
	xgb_model = xgb.XGBClassifier(
	learning_rate =0.1,
	n_estimators=1000,
	max_depth=5,
	min_child_weight=1,
	gamma=0,
	subsample=0.8,
	colsample_bytree=0.8,
	objective= 'binary:logistic',
	nthread=4,
	scale_pos_weight=1,
	seed=27,
	eval_metric=metrics)
	xgb_param = xgb_model.get_xgb_params()
	dtrain = xgb.DMatrix(X_train, label=Y_train)
	cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=xgb_model.get_params()['n_estimators'], nfold=10,
	metrics=metrics, early_stopping_rounds=50, shuffle=True, verbose_eval=False)

	xgb_model.set_params(n_estimators=cvresult.shape[0])
	para_test = {
	'max_depth':range(3,11),
	'min_child_weight':range(1,13)
	}
	gsearch1 = GridSearchCV(estimator=xgb_model,param_grid=para_test, scoring=scoring, n_jobs=1,iid=False,cv=5,verbose=0)
	gsearch1.fit(X_train,Y_train)
	best_params =  gsearch1.best_params_
	best_score = gsearch1.best_score_
	xgb_model.set_params(max_depth=best_params['max_depth'])
	xgb_model.set_params(min_child_weight=best_params['min_child_weight'])

	para_test = {
	'gamma':[i/10.0 for i in range(0,6)]
	}
	gsearch2 = GridSearchCV(estimator=xgb_model,param_grid=para_test, scoring=scoring,n_jobs=1,iid=False,cv=5,verbose=0)
	gsearch2.fit(X_train,Y_train)
	best_params =  gsearch2.best_params_
	best_score = gsearch2.best_score_
	xgb_model.set_params(gamma=best_params['gamma'])

	para_test = {
	'subsample':[i/10.0 for i in range(4,11)],
	'colsample_bytree':[i/10.0 for i in range(6,11)]
	}
	gsearch3 = GridSearchCV(estimator=xgb_model,param_grid=para_test, scoring=scoring,n_jobs=1,iid=False,cv=5,verbose=0)
	gsearch3.fit(X_train,Y_train)
	best_params =  gsearch3.best_params_
	best_score = gsearch3.best_score_
	xgb_model.set_params(subsample=best_params['subsample'])
	xgb_model.set_params(colsample_bytree=best_params['colsample_bytree'])

	para_test = {
	'reg_alpha':[0, 1e-5, 5e-5, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1, 5, 100]
	}
	gsearch4 = GridSearchCV(estimator=xgb_model,param_grid=para_test, scoring=scoring, n_jobs=1,iid=False,cv=5,verbose=0)
	gsearch4.fit(X_train,Y_train)
	best_params =  gsearch4.best_params_
	best_score = gsearch4.best_score_
	xgb_model.set_params(reg_alpha=best_params['reg_alpha'])

	para_test = {
	'reg_lambda':[0, 1e-5, 5e-5, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1, 5, 100]
	}
	gsearch5 = GridSearchCV(estimator=xgb_model,param_grid=para_test, scoring=scoring, n_jobs=1,iid=False,cv=5,verbose=0)
	gsearch5.fit(X_train,Y_train)
	best_params =  gsearch5.best_params_
	best_score = gsearch5.best_score_
	xgb_model.set_params(reg_lambda=best_params['reg_lambda'])


	if plot == True:
		paratune = ['max_depth min_child_weight', 'gamma', 'subsample colsample_bytree', 'reg_alpha', 'reg_lambda']
		for n, gsearch in enumerate([gsearch1, gsearch2, gsearch3, gsearch4, gsearch5]):
			Ytrain_pred = gsearch.predict(X_train)
			Ytrain_pred_prob  = gsearch.predict_proba(X_train)
			Ytrain_pred = (Ytrain_pred>0.5)*1
			print(paratune[n])
			#print('train_acc: ', accuracy_score(Y_train,Ytrain_pred) )
			#print('train_f1: ', f1_score(Y_train,Ytrain_pred,average='binary'))
			#print('roc_auc_score: ', roc_auc_score(Y_train, Ytrain_pred_prob[:,1]))
			print('train loss: ', log_loss(Y_train, Ytrain_pred_prob))
			print('best val loss: ', gsearch.best_score_*-1)
			#input()
	return xgb_model
