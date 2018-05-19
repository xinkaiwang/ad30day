#!/usr/bin/env python

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# %matplotlib inline

SEED = 222
np.random.seed(SEED)

df = pd.read_csv('input.csv')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def get_train_test(test_size=0.95) :
	"""Split data into train and test sets."""
	y = 1 * (df.cand_pty_affiliation == "REP")
	X = df.drop(['cand_pty_affiliation'], axis=1)
	X = pd.get_dummies(X) # sparse=True
	X.drop(X.columns[X.std() == 0], axis = 1, inplace = True)
	return train_test_split(X, y, test_size = test_size, random_state=SEED)

xtrain, xtest, ytrain, ytest = get_train_test()

print("\nExample data:")
print(df.head())

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline

def get_models():
	"""Generate a library of base learners."""
	nb = GaussianNB()
	svc = SVC(C=100, probability=True)
	knn = KNeighborsClassifier(n_neighbors=3)
	lr = LogisticRegression(C=100, random_state=SEED)
	nn = MLPClassifier((80,10), early_stopping=False, random_state=SEED)
	gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
	rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=SEED)

	models = {'svm':svc,
		'knn': knn,
		'naive bayes': nb,
		'mlp-nn': nn,
		'random forest': rf,
		'gbm': gb,
		'logistic': lr,
		}
	return models

def train_predict(model_list) :
	"""Fit models in list on trainning set and return preds"""
	P = np.zeros((ytest.shape[0], len(model_list)))
	P = pd.DataFrame(P)

	print("Fitting models")
	cols = list()
	for i, (name, m) in enumerate(model_list.items()):
		print("%s..." % name, end=" ", flush=False)
		m.fit(xtrain, ytrain)
		P.iloc[:, i] = m.predict_proba(xtest)[:, 1]
		cols.append(name)
		print("done")

	P.columns = cols
	print("Done. \n")
	return P

def score_models(P,y):
	"""Score model in prediction DF"""
	print("Scoring models")
	for m in P.columns:
		score = roc_auc_score(y, P.loc[:,m])
		print("%-26s: %.3f" % (m, score))
	print ("Done.\n")

models = get_models()
P = train_predict(models)

from mlens.visualization import corrmat

corrmat(P.corr(), inflate=False)
plt.show()

print("Ensemble ROC-AUC score: %.3f" % roc_auc_score(ytest, P.mean(axis=1)))

from sklearn.metrics import roc_curve

def plot_roc_curve(ytest, P_base_learners, P_ensemble, labels, ens_label):
	"""Plot the roc curve fro base learners and ensemble."""
	plt.figure(figsize=(10,8))
	plt.plot([0,1], [0,1], 'k--')

	cm = [plt.cm.rainbow(i) for i in np.linspace(0, 1.0, P_base_learners.shape[1] + 1)]
	for i in range(P_base_learners.shape[1]):
		p = P_base_learners[:, i]
		fpr, tpr, _ = roc_curve(ytest, p)
		plt.plot(fpr, tpr, label=labels[i], c=cm[i+1])

	fpr, tpr, _ = roc_curve(ytest, P_ensemble)
	plt.plot(fpr, tpr, label=ens_label, c=cm[0])
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.legend(frameon=False)
	plt.show()

plot_roc_curve(ytest, P.values, P.mean(axis=1), list(P.columns), "ensemble")

p = P.apply(lambda x: 1*(x>=0.5).value_counts(normalize=True))
p.index = ["DEM", "REP"]
p.loc["REP", :].sort_values().plot(kind="bar")
plt.axhline(0.25, color="k", linewidth=0.5)
plt.text(0., 0.23, "True share republicans")
plt.show()


