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

	FEATURE_NAMES = [value for key, value in Columns.__dict__.items() if key.isupper()][:-1]

	def __init__(self):
		self.classifier = None
		self.features = None
		self.values = None

	def set_training_dataframe(self, dataframe):
		self.features = dataframe[self.FEATURE_NAMES]
		self.values = dataframe[[self.Columns.SALARY]]
		self.classifier = None

	def train(self):
		if self.classifier is None:
			self.classifier = GaussianNB()
			self.classifier.fit(self.features, self.values)

	def predict(self, features):
		if self.classifier is None:
			self.train()
		return self.classifier.predict(features)


class SalaryModelValidator():
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

	def __init__(self):
		self.model = SalaryModel()
		self.test_dataframe = None
		self.training_dataframe = None

	@classmethod
	def prep_dataframe(cls, dataset_file):
		dataframe = pd.read_csv(dataset_file, header=None, names=cls.COLUMNS)
		for column in cls.COLUMNS:
			dataframe[column] = dataframe[column].replace(' ?', np.nan)
		dataframe = dataframe.dropna()
		for dimension, feature in SalaryModel.Columns.__dict__.items():
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

		predicted = self.model.predict(self.test_dataframe[SalaryModel.FEATURE_NAMES])
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
