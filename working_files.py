import pandas as pd



def get_train_test_sub():

    # -999.0 == NaN
    # PRI_jet_num is an integer
    # Everything else should be a float


    df_train = pd.read_csv('../higgs-boson/data/training.csv')
    df_test = pd.read_csv('../higgs-boson/data/test.csv')
    df_sub = pd.read_csv('../higgs-boson/data/random_submission.csv')

    return df_train, df_test, df_sub


def create_submission_file(fn, df, labels=None):
    outputfile = open(fn, "w")
    outputfile.write("EventId,RankOrder,Class\n")

    if labels:
        df['Label'] = labels

    ind_id = list(df.columns).index('EventId')
    ind_lbl = list(df.columns).index('EventId')

    rnk = 1
    for row in df.values:
        outputfile.write(str(row[ind_id]) + ",")
        outputfile.write(str(rnk) + ",")
        outputfile.write(str(row[ind_lbl]) + "\n")
        rnk += 1

    outputfile.close()
