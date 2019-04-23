from collections import namedtuple
import random

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

PRINT = False


class SalaryModel():
    class Columns:
        AGE = 'age'
        WORKCLASS = 'workclass'
        FNLWGT = 'fnlwgt'
        EDUCATION = 'education'
        EDUCATION_NUM = 'education_num'
        MARITAL_STATUS = 'marital_status'
        OCCUPATION = 'occupation'
        RELATIONSHIP = 'relationship'
        RACE = 'race'
        SEX = 'sex'
        CAPITAL_GAIN = 'capital_gain'
        CAPITAL_LOSS = 'capital_loss'
        HOURS_PER_WEEK = 'hours_per_week'
        NATIVE_COUNTRY = 'native_country'
        SALARY = 'salary'

    IRRELEVANT_COLUMNS = [Columns.WORKCLASS, Columns.EDUCATION, Columns.OCCUPATION]
    FEATURE_NAMES = [value for key, value in Columns.__dict__.items() if key.isupper()][:-1]

    def __init__(self):
        self.classifier = None
        self.features = None
        self.values = None
        self.used_features = [feature for feature in self.FEATURE_NAMES if feature not in self.IRRELEVANT_COLUMNS]

    def set_training_dataframe(self, dataframe):
        self.features = dataframe[self.used_features]
        self.values = dataframe[[self.Columns.SALARY]]
        self.classifier = None

    def train(self):
        if self.classifier is None:
            self.classifier = GaussianNB()
            self.classifier.fit(self.features, self.values.values.ravel())

    def predict(self, features):
        if self.classifier is None:
            self.train()
        return self.classifier.predict(features)


class SalaryModelValidator():
    COLUMNS = SalaryModel.FEATURE_NAMES + [SalaryModel.Columns.SALARY]
    SALARY_MAP = {' <=50K': 0, ' >50K': 1, ' <=50K.': 0, ' >50K.': 1}
    MARITAL_STATUS_MAP = {' Never-married': 0, ' Married-civ-spouse': 1, ' Divorced': 0,
                          ' Married-spouse-absent': 1, ' Separated': 0, ' Married-AF-spouse': 1,
                          ' Widowed': 0}
    RELATIONSHIP_MAP = {' Not-in-family': 0, ' Husband': 1, ' Wife': 1, ' Own-child': 0, ' Unmarried': 1, ' Other-relative': 0}
    RACE_MAP = {' White': 3, ' Black': 2, ' Asian-Pac-Islander': 4, ' Amer-Indian-Eskimo': 1, ' Other': 0}
    SEX_MAP = {' Male': 1, ' Female': 0}
    NATIVE_COUNTRY_MAP = {value: 1 if value == ' United-States' else 0
                            for value in [' United-States', ' Cuba', ' Jamaica', ' India', ' Mexico',
                           ' Puerto-Rico', ' Honduras', ' England', ' Canada', ' Germany',
                           ' Iran', ' Philippines', ' Poland', ' Columbia', ' Cambodia',
                           ' Thailand', ' Ecuador', ' Laos', ' Taiwan', ' Haiti', ' Portugal',
                           ' Dominican-Republic', ' El-Salvador', ' France', ' Guatemala',
                           ' Italy', ' China', ' South', ' Japan', ' Yugoslavia', ' Peru',
                           ' Outlying-US(Guam-USVI-etc)', ' Scotland', ' Trinadad&Tobago',
                           ' Greece', ' Nicaragua', ' Vietnam', ' Hong', ' Ireland',
                           ' Hungary', ' Holand-Netherlands']}

    def __init__(self):
        self.model = SalaryModel()
        self.test_dataframe = None
        self.training_dataframe = None

    @classmethod
    def prep_dataframe(cls, dataset_file):
        dataframe = pd.read_csv(dataset_file, header=None, names=cls.COLUMNS)
        dataframe.drop(labels=SalaryModel.IRRELEVANT_COLUMNS, axis=1, inplace=True)
        for column in cls.COLUMNS:
            if column in SalaryModel.IRRELEVANT_COLUMNS:
                continue
            dataframe[column] = dataframe[column].replace(' ?', np.nan)
        dataframe = dataframe.dropna()
        for dimension, feature in SalaryModel.Columns.__dict__.items():
            if feature in SalaryModel.IRRELEVANT_COLUMNS:
                continue
            value_map = getattr(cls, '{}_MAP'.format(dimension), None)
            if value_map:
                dataframe[feature] = dataframe[feature].map(value_map).astype(int)
        return dataframe

    def load_data(self, training_dataset_file, test_dataset_file):    
        self.training_dataframe = self.prep_dataframe(training_dataset_file)
        self.test_dataframe = self.prep_dataframe(test_dataset_file)

    def validate(self):
        self.model.set_training_dataframe(self.training_dataframe)
        self.model.train()

        predicted = self.model.predict(self.test_dataframe[self.model.used_features])
        actual = self.test_dataframe[[SalaryModel.Columns.SALARY]]
        
        return accuracy_score(actual, predicted)

    @classmethod
    def run(cls, training_dataset_file, test_dataset_file):
        validator = cls()
        validator.load_data(training_dataset_file, test_dataset_file)
        print('Accuracy: {}'.format(validator.validate()))

def run(training_dataset_file, test_dataset_file):
    SalaryModelValidator.run(training_dataset_file, test_dataset_file)


if __name__ == '__main__':
    run('adult.data', 'adult.test')
