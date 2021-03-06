"""
Naive Bayes Classifier using UCI Adult dataset.

This implementation doesn't do any actual data pre-processing, it just assigns
random values to string-based features.
"""

from collections import namedtuple
import random

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd


class SalaryModel():
    """
    Wrapper around Gaussian Naive Bayes classifier providing convenience
    methods for working with Adult dataset-based dataframes.
    """

    class Columns:
        """
        List of columns and their identifiers in order of appearance in adult dataframe.

        We don't want to use string literals around. It's very error-prone.
        """

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

    # Use all columns in order, skipping the salary which is the result.
    FEATURE_NAMES = [value for key, value in Columns.__dict__.items() if key.isupper()][:-1]

    def __init__(self):
        """
        Initialize variables used later on.
        """

        self.classifier = None
        self.features = None
        self.values = None

    def set_training_dataframe(self, dataframe):
        """
        Set data points (features) and results (values) according to input dataframe.

        Reset the classifier to ensure it will be retrained before next prediction.
        """

        self.features = dataframe[self.FEATURE_NAMES]
        self.values = dataframe[[self.Columns.SALARY]]
        self.classifier = None

    def train(self):
        """
        Train the classifier using current state of feratures and values.
        """

        if self.classifier is None:
            self.classifier = GaussianNB()
            self.classifier.fit(self.features, self.values.values.ravel())

    def predict(self, features):
        """
        Train the classifier if needed and predict the result for given array of features.
        """

        if self.classifier is None:
            self.train()
        return self.classifier.predict(features)


class SalaryModelDatasetProcessor():
    """
    Provides method (prep_dataframe) reading from a file in accordance to adult data set
    and returning a dataframe ready to use in a model.

    Additionally, contains mappings describing the process of conversion from string literal
    data points to integers.

    The integer values are assigned in order of appearance in the data set.
    """

    COLUMNS = SalaryModel.FEATURE_NAMES + [SalaryModel.Columns.SALARY]

    SALARY_MAP = {' <=50K': 0, ' >50K': 1, ' <=50K.': 0, ' >50K.': 1}
    WORKCLASS_MAP = {' State-gov': 0, ' Self-emp-not-inc': 1, ' Private': 2, ' Federal-gov': 3,
                     ' Local-gov': 4, ' Self-emp-inc': 5, ' Without-pay': 6}
    EDUCATION_MAP = {' Bachelors': 0, ' HS-grad': 1, ' 11th': 2, ' Masters': 3, ' 9th': 4,
                     ' Some-college': 5, ' Assoc-acdm': 6, ' 7th-8th': 7, ' Doctorate': 8,
                     ' Assoc-voc': 9, ' Prof-school': 10, ' 5th-6th': 11, ' 10th': 12, ' Preschool': 13,
                     ' 12th': 14, ' 1st-4th': 15}
    MARITAL_STATUS_MAP = {' Never-married': 0, ' Married-civ-spouse': 1, ' Divorced': 2,
                          ' Married-spouse-absent': 3, ' Separated': 4, ' Married-AF-spouse': 5,
                          ' Widowed': 6}
    OCCUPATION_MAP = {' Adm-clerical': 0, ' Exec-managerial': 1, ' Handlers-cleaners': 2,
                      ' Prof-specialty': 3, ' Other-service': 4, ' Sales': 5, ' Transport-moving': 6,
                      ' Farming-fishing': 7, ' Machine-op-inspct': 8, ' Tech-support': 9,
                      ' Craft-repair': 10, ' Protective-serv': 11, ' Armed-Forces': 12,
                      ' Priv-house-serv': 13}
    RELATIONSHIP_MAP = {' Not-in-family': 0, ' Husband': 1, ' Wife': 2, ' Own-child': 3, ' Unmarried': 4, ' Other-relative': 5}
    RACE_MAP = {' White': 0, ' Black': 1, ' Asian-Pac-Islander': 2, ' Amer-Indian-Eskimo': 3, ' Other': 4}
    SEX_MAP = {' Male': 0, ' Female': 1}
    NATIVE_COUNTRY_MAP = {value: index for index, value in enumerate([' United-States', ' Cuba', ' Jamaica', ' India', ' Mexico',
                           ' Puerto-Rico', ' Honduras', ' England', ' Canada', ' Germany',
                           ' Iran', ' Philippines', ' Poland', ' Columbia', ' Cambodia',
                           ' Thailand', ' Ecuador', ' Laos', ' Taiwan', ' Haiti', ' Portugal',
                           ' Dominican-Republic', ' El-Salvador', ' France', ' Guatemala',
                           ' Italy', ' China', ' South', ' Japan', ' Yugoslavia', ' Peru',
                           ' Outlying-US(Guam-USVI-etc)', ' Scotland', ' Trinadad&Tobago',
                           ' Greece', ' Nicaragua', ' Vietnam', ' Hong', ' Ireland',
                           ' Hungary', ' Holand-Netherlands'])}

    @classmethod
    def prep_dataframe(cls, dataset_file):
        """
        Prepare the dataframe suitable for SalaryModel from input dataset file.

        Drops rows with unknown feature values and maps string literals to ints.
        """

        dataframe = pd.read_csv(dataset_file, header=None, names=cls.COLUMNS)
        
        for column in cls.COLUMNS:
            dataframe[column] = dataframe[column].replace(' ?', np.nan)
        dataframe = dataframe.dropna()

        for dimension, feature in SalaryModel.Columns.__dict__.items():
            value_map = getattr(cls, '{}_MAP'.format(dimension), None)
            if value_map:
                dataframe[feature] = dataframe[feature].map(value_map).astype(int)
        
        return dataframe


class SalaryModelValidator():
    """
    Provide validataion capabilities for SalaryModel.

    See run for entrypoint.
    """

    dataset_processor = SalaryModelDatasetProcessor

    def __init__(self):
        """
        Initialize variables used later on.
        """

        self.model = SalaryModel()
        self.test_dataframe = None
        self.training_dataframe = None

    @classmethod
    def prep_dataframe(cls, dataset_file):
        """
        Proxy method for dataset_processor's prep_dataframe.
        """

        return cls.dataset_processor.prep_dataframe(dataset_file)

    def load_data(self, training_dataset_file, test_dataset_file):
        """
        Set validator instance's dataframes to result of processing input files.
        """

        self.training_dataframe = self.prep_dataframe(training_dataset_file)
        self.test_dataframe = self.prep_dataframe(test_dataset_file)

    def validate(self):
        """
        Validate test dataset against model trained with training dataset.
        """

        self.model.set_training_dataframe(self.training_dataframe)
        self.model.train()

        predicted = self.model.predict(self.test_dataframe[SalaryModel.FEATURE_NAMES])
        actual = self.test_dataframe[[SalaryModel.Columns.SALARY]]
        
        return accuracy_score(actual, predicted)

    @classmethod
    def run(cls, training_dataset_file, test_dataset_file):
        """
        Load data and run validation using input datasets.
        
        Print accuracy.
        """
        validator = cls()
        validator.load_data(training_dataset_file, test_dataset_file)
        print('Accuracy: {:2.2f}%'.format(validator.validate() * 100))


if __name__ == '__main__':
    SalaryModelValidator.run('adult.data', 'adult.test')
