#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline

def get_models(randomSeed):
	"""Generate a library of base learners."""
	nb = GaussianNB()
	svc = SVC(C=100, probability=True)
	knn = KNeighborsClassifier(n_neighbors=3)
	lr = LogisticRegression(C=100, random_state=randomSeed)
	nn = MLPClassifier((80,10), early_stopping=False, random_state=randomSeed)
	gb = GradientBoostingClassifier(n_estimators=100, random_state=randomSeed)
	rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=randomSeed)

	models = {'svm':svc,
		'knn': knn,
		'naive bayes': nb,
		'mlp-nn': nn,
		'random forest': rf,
		'gbm': gb,
		'logistic': lr,
		}
	return models


def get_train_test(random_state, test_size=0.95):
	"""Split data into train and test sets."""
	df = pd.read_csv('input.csv')
	y = 1 * (df.cand_pty_affiliation == "REP")
	X = df.drop(['cand_pty_affiliation'], axis=1)
	X = pd.get_dummies(X) # sparse=True
	X.drop(X.columns[X.std() == 0], axis = 1, inplace = True)
	return train_test_split(X, y, test_size = test_size, random_state=random_state)

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

