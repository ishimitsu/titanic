import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from functools import partial
import optuna
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

train_csv = 'input/train.csv'
test_csv = 'input/test.csv'
submission_temp = 'input/gender_submission.csv'

# kfold params
fold = 4
random_state = 71

search_hyper_param = False
lgbm_params = {
    'objective': 'binary',  # default only 82.83
    'max_bin': 300,
    'learning_rate': 0.5,  # default 0.1
    'num_leaves': 21,  # default 31
}


def label_encorder (data, label_list: list):
    for l in label_list:
        le = LabelEncoder()
        data[l] = le.fit_transform(data[l].fillna('NA'))  # fillna erase NaN data
    return data


def objective(train_x, train_y, fold, random_state, trial):

    kf = KFold(n_splits=fold, shuffle=True, random_state=random_state)
    lgbm_params = {
        'objective': 'binary',
        'max_bin': trial.suggest_int('max_bin', 255, 500),
        'learning_rate': 0.05,
        'num_leaves': trial.suggest_int('num_leaves', 32, 128),
    }

    score_list = []
    for fold_, (tr_idx, va_idx) in enumerate(kf.split(train_x, train_y)):
        # separate train-data to train/validation data.
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # covert to lgb.Dataset
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_valid = lgb.Dataset(va_x, va_y)

        evals_result = {}
        # training
        gbm = lgb.train(params=lgbm_params,
                        train_set=lgb_train,
                        valid_sets=[lgb_train, lgb_valid],
                        early_stopping_rounds=20,
                        evals_result=evals_result,
                        verbose_eval=10);
        va_y_pred_valid = gbm.predict(va_x)
        score_list.append(log_loss(va_y, va_y_pred_valid))

    return round(np.mean(score_list), 2)


def fit(train_x, train_y, kf, lgbm_params):
    models = []
    score_list = []

    for fold_, (tr_idx, va_idx) in enumerate(kf.split(train_x, train_y)):
        # separate train-data to train/validation data.
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # covert to lgb.Dataset
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_valid = lgb.Dataset(va_x, va_y)

        evals_result = {}
        # training
        gbm = lgb.train(params=lgbm_params,
                        train_set=lgb_train,
                        valid_sets=[lgb_train, lgb_valid],
                        early_stopping_rounds=20,
                        evals_result=evals_result,
                        verbose_eval=10);

        oof = (gbm.predict(va_x) > 0.5).astype(int)  # covert score to 0 or 1
        score_list.append(round(accuracy_score(va_y, oof)*100,2))  # cal accuracy and put list
        models.append(gbm)  # put trained-model
        print(f'fold{fold_ + 1} end\n' )

    avg_score = round(np.mean(score_list), 2)
    print(score_list, "avg score ", avg_score)
    return models, avg_score


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

    if search_hyper_param:
        study_params = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        f = partial(objective, train_x, train_y, fold, random_state)
        study_params.optimize(f, n_trials=10)
        print("Optuna found params: ", study_params.best_params)
        lgbm_params['max_bin'] = study_params.best_params['max_bin']
        lgbm_params['num_leaves'] = study_params.best_params['num_leaves']

    # kf = KFold(n_splits=fold, shuffle=True, random_state=random_state)
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    models, score = fit(train_x, train_y, kf, lgbm_params)

    # create submission
    test_pred = np.zeros((len(test_x), fold))  # create prediction-data for test data (418xfold)
    for fold_, gbm in enumerate(models):
        test_pred[:, fold_] = gbm.predict(test)

    pred = (np.mean(test_pred, axis=1) > 0.5).astype(int)
    sample_submission['Survived'] = pred
    submission_csv = f'{fold}-fold_lgbm.csv'
    sample_submission.to_csv(submission_csv, index=False)


main()
