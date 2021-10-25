import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

train_csv = 'input/train.csv'
test_csv = 'input/test.csv'
submission_csv = 'input/gender_submission.csv'


def label_encorder (data, label_list: list):
    for l in label_list:
        le = LabelEncoder()
        data[l] = le.fit_transform(data[l].fillna('NA'))
    return data


def main():
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    sample_submission = pd.read_csv(submission_csv)
    print(train.head())

    train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'Survived'], axis=1, inplace=True)
    test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
    print(train.head())

    train = label_encorder(train, ['Sex', 'Embarked'])
    test = label_encorder(test, ['Sex', 'Embarked'])
    print(train.head())





main()
