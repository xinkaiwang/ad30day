#!/usr/bin/env python

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#%matplotlib inline

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

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
	n_estimators = 10,
	max_features = 3,
	random_state = SEED,
)

rf.fit(xtrain, ytrain)
p = rf.predict_proba(xtest)[:, 1]

print("Average of decision tree ROC-AUC store: %.3f" % roc_auc_score(ytest,p))

