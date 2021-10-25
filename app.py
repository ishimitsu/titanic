import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

train_csv = 'input/train.csv'
test_csv = 'input/test.csv'
submission_csv = 'input/gender_submission.csv'


def label_encorder (data, label_list: list):
    for l in label_list:
        le = LabelEncoder()
        data[l] = le.fit_transform(data[l].fillna('NA'))  # fillna erase NaN data
    return data


def preprocess_train(train, test, obj_val: str, label_encord_target: list, drop_cols: list):
    # separate features and obj-val
    train_x = train.drop([obj_val], axis=1)
    train_y = train[obj_val]
    test_x = test.copy()

    # label encording
    train_x = label_encorder(train_x, label_encord_target)
    test_y = label_encorder(test, label_encord_target)

    # Remove unneccessary features
    train_x.drop(drop_cols, axis=1, inplace=True)
    test_x.drop(drop_cols, axis=1, inplace=True)

    return train_x, train_y, test_x


def main():
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    sample_submission = pd.read_csv(submission_csv)
    print(train.head())

    obj_val = 'Survived'
    label_encord_target = ['Sex', 'Embarked']
    drop_cols = ['PassengerId', 'Name', 'Cabin', 'Ticket']
    train_x, train_y, test_x = preprocess_train(train, test, obj_val, label_encord_target, drop_cols)
    print(train_x.head())

    score_list = []
    models = []
    kf = KFold(n_splits=4, shuffle=True, random_state=71)

 #  for tr_idx, va_idx in kf.split(train_x):
    for fold_, (tr_idx, va_idx) in enumerate(kf.split(train_x, train_y)):
        # separate train-data to train/validation data.
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # covert to lgb.Dataset
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_valid = lgb.Dataset(va_x, va_y)

        # set lgbm params
        lgbm_params = {'objective': 'binary'}
        # training
        evals_result = {}
        gbm = lgb.train(params = lgbm_params,
                        train_set = lgb_train,
                        valid_sets= [lgb_train, lgb_valid],
                        early_stopping_rounds=20,
                        evals_result=evals_result,
                        verbose_eval=10);

        oof = (gbm.predict(va_x) > 0.5).astype(int)
        score_list.append(round(accuracy_score(va_y, oof)*100,2))
        models.append(gbm)  # put trained-model
        print(f'fold{fold_ + 1} end\n' )
    print(score_list, "avg score ", round(np.mean(score_list), 2))


main()
