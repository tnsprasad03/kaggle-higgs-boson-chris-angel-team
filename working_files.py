import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import cross_val_score

def get_train_test():
    df_train = pd.read_csv('../higgs-boson-data/training.csv',
                            na_values=-999.0,
                            engine='c')
    df_test = pd.read_csv('../higgs-boson-data/test.csv',
                            na_values=-999.0,
                            engine='c')

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

clf = LR(penalty='l2')
print cross_val_score(clf, X_train, Y_train, scoring="accuracy", cv=3)

clf = LR(penalty='l2')
print cross_val_score(clf, X_train, Y_train, scoring="accuracy", cv=3)

