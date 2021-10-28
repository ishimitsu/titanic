import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

train_csv = 'input/train.csv'
test_csv = 'input/test.csv'
submission_temp = 'input/gender_submission.csv'
submission_csv = '4-fold_lgbm.csv'


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
    test_x = label_encorder(test, label_encord_target)

    # Remove unneccessary features
    train_x.drop(drop_cols, axis=1, inplace=True)
    test_x.drop(drop_cols, axis=1, inplace=True)

    return train_x, train_y, test_x


def main():
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    sample_submission = pd.read_csv(submission_temp)
    obj_val = 'Survived'

    # Create new features that check
    data = pd.concat([train, test], sort=False)
    data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
    train['FamilySize'] = data['FamilySize'][:len(train)]
    test['FamilySize'] = data['FamilySize'][len(train):]

    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
    train['IsAlone'] = data['IsAlone'][:len(train)]
    test['IsAlone'] = data['IsAlone'][len(train):]

    label_encord_target = ['Sex', 'Embarked']
    drop_cols = ['PassengerId', 'Name', 'Cabin', 'Ticket']
    train_x, train_y, test_x = preprocess_train(train, test, obj_val, label_encord_target, drop_cols)
    print(train_x.head())

    score_list = []
    models = []
    fold = 4
    random_state = 71
    kf = KFold(n_splits=fold, shuffle=True, random_state=random_state)

    early_stopping_rounds = 20
    verbose_eval = 10

    for fold_, (tr_idx, va_idx) in enumerate(kf.split(train_x, train_y)):
        # separate train-data to train/validation data.
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # covert to lgb.Dataset
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_valid = lgb.Dataset(va_x, va_y)
        # lgbm params
        lgbm_params = {'objective': 'binary'}
        evals_result = {}
        # training
        gbm = lgb.train(params=lgbm_params,
                        train_set=lgb_train,
                        valid_sets=[lgb_train, lgb_valid],
                        early_stopping_rounds=early_stopping_rounds,
                        evals_result=evals_result,
                        verbose_eval=verbose_eval);

        oof = (gbm.predict(va_x) > 0.5).astype(int)  # covert score to 0 or 1
        score_list.append(round(accuracy_score(va_y, oof)*100,2))  # cal accuracy and put list
        models.append(gbm)  # put trained-model
        print(f'fold{fold_ + 1} end\n' )
    print(score_list, "avg score ", round(np.mean(score_list), 2))


    # create submission
    test_pred = np.zeros((len(test_x), fold))  # create prediction-data for test data (418xfold)
    for fold_, gbm in enumerate(models):
        test_pred[:, fold_] = gbm.predict(test)

    pred = (np.mean(test_pred, axis=1) > 0.5).astype(int)
    sample_submission['Survived'] = pred
    sample_submission.to_csv(submission_csv, index=False)


main()
