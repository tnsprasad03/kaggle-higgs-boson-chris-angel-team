import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import cross_val_score
from pprint import pprint
from time import time
import logging
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
import pandas as pd
import numpy as np
import json
import sys


def get_train_test():
    df_train = pd.read_csv('data/training.csv',na_values=-999.0,engine='c')
    df_test = pd.read_csv('data/test.csv', na_values=-999.0, engine='c')

    # fillna with mean values:
    df_train =  df_train.fillna(df_train.mean())
    df_test = df_test.fillna(df_test.mean())

    return df_train, df_test

def get_X_Y(df, Ycol='Label'):
    X = df.copy()
    if Ycol in list(df.columns):
        X = X.drop(Ycol, 1)
        X = X.drop('Weight', 1)
        Y = [1 if z == 's' else 0 for z in df[Ycol]]
        Y = np.array(Y).reshape(X.shape[0],1).ravel()
        return X, Y
    else:
        return X


def create_submission_file(df, fn='submission.csv', labels=None):
    outputfile = open(fn, "w")
    outputfile.write("EventId,RankOrder,Class\n")

    if labels: df['Label'] = labels

    ind_id = list(df.columns).index('EventId')
    ind_lbl = list(df.columns).index('EventId')

    rnk = 1
    for row in df.values:
        outputfile.write(str(row[ind_id]) + ",")
        outputfile.write(str(rnk) + ",")
        outputfile.write(str(row[ind_lbl]) + "\n")
        rnk += 1

    outputfile.close()


df_train, df_test = get_train_test()

# X, Y where X is df and Y is a numpy array
X_train, Y_train = get_X_Y(df_train)
X_test = get_X_Y(df_test)
features = X_train.columns
X_train = X_train.values
X_test = X_test.values

print df_train.corr()
#sys.exit(0)

pipeline = Pipeline([
    # ('vect', CountVectorizer()),
    # ('tfidf', TfidfTransformer()),
    ('clf', svm.LinearSVC())
])

parameters = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # #'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__C': (0.001,0.1, 1.0,10,100),
    #'clf__penalty': ('l2'),
    #'clf__degree': (4,5),
    #'clf__n_iter': (10, 50, 80),
}

#sys.exit(0)
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring="roc_auc", cv=5)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
grid_search.fit(X_train, Y_train)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


clf = LR(penalty='l2')
print cross_val_score(clf, X_train, Y_train, scoring="accuracy", cv=3)

clf = LR(penalty='l2')
print cross_val_score(clf, X_train, Y_train, scoring="accuracy", cv=3)

