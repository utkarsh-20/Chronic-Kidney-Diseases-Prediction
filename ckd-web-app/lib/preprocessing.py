import pandas as pd
from sklearn.externals import joblib


def create_df():
    cols = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc',
            'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    return pd.DataFrame(columns=cols, index=[0])


def fix_missing(df, na_dict):
    df.replace(na_dict, inplace=True)


def load_scaler():
    return joblib.load('./scaler/min-max.pkl')


def load_model():
    return joblib.load('./model/rf-ckd.pkl')
