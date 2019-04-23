from collections import namedtuple
import random

from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as colors
import numpy as np
import pandas as pd

PRINT = False

Heatmap = namedtuple('Heatmap', ['predicted', 'actual', 'error'])
Validation = namedtuple('Validation', ['predicted', 'actual', 'error', 'heatmap'])


class MidpointNormalize(colors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


class GeoModel():
	class Columns:
		OSM_ID = 'osm_id'
		LATITUDE = 'latitude'
		LONGTITUDE = 'longtitude'
		ALTITUDE = 'altitude'

	def __init__(self, k_neighbors):
		self.k_neighbors = k_neighbors
		self.regressor = None
		self.features = None
		self.values = None

	def set_training_dataframe(self, dataframe):
		self.features = dataframe[[self.Columns.LATITUDE, self.Columns.LONGTITUDE]]
		self.values = dataframe[[self.Columns.ALTITUDE]]
		self.regressor = None

	def train(self):
		if self.regressor is None:
			self.regressor = KNeighborsRegressor(self.k_neighbors)
			self.regressor.fit(self.features, self.values)

	def predict(self, features):
		if self.regressor is None:
			self.train()
		return self.regressor.predict(features)


class GeoModelValidator():
	COLUMNS = [GeoModel.Columns.OSM_ID, GeoModel.Columns.LATITUDE, GeoModel.Columns.LONGTITUDE, GeoModel.Columns.ALTITUDE]

	def __init__(self, k_fold, k_neighbors, k_iterations):
		self.k_fold = k_fold
		self.k_iterations = k_iterations
		self.model = GeoModel(k_neighbors)
		self.print = print

	def load_data(self, dataset_file):
		dataframe = pd.read_csv(dataset_file, header=None, names=self.COLUMNS)
		to_normalize = dataframe[[GeoModel.Columns.LATITUDE, GeoModel.Columns.LONGTITUDE]]
		scaler = MinMaxScaler()
		normalized = scaler.fit_transform(to_normalize)
		self.dataframe = pd.DataFrame(normalized, columns=[GeoModel.Columns.LATITUDE, GeoModel.Columns.LONGTITUDE])
		self.dataframe[GeoModel.Columns.ALTITUDE] = dataframe[GeoModel.Columns.ALTITUDE]

	def validate(self):
		if PRINT:
			print('MIN: {}, MAX: {}'.format(self.dataframe[GeoModel.Columns.ALTITUDE].min(),
											self.dataframe[GeoModel.Columns.ALTITUDE].max()))
		for _ in range(self.k_iterations):
			predicted, actual, heatmap = self.validate_fold()
			yield Validation(predicted, actual, mean_absolute_error(actual, predicted), heatmap)

	def validate_fold(self):
		test_rows = random.sample(self.dataframe.index.tolist(), int(len(self.dataframe) / self.k_fold))
		training_rows = set(range(len(self.dataframe))) - set(test_rows)

		test_dataframe = self.dataframe.iloc[test_rows]
		training_dataframe = self.dataframe.drop(test_rows)
		
		self.model.set_training_dataframe(training_dataframe)
		self.model.train()

		predicted = self.model.predict(test_dataframe[[GeoModel.Columns.LATITUDE, GeoModel.Columns.LONGTITUDE]])
		actual = test_dataframe[[GeoModel.Columns.ALTITUDE]]

		predicted_df = test_dataframe.copy()
		predicted_df[GeoModel.Columns.ALTITUDE] = predicted

		error_df = test_dataframe.copy()
		error_df[GeoModel.Columns.ALTITUDE] = [predicted[index][0] - actual.iloc[index][GeoModel.Columns.ALTITUDE]
				 for index in range(len(actual))]

		return predicted, actual, Heatmap(predicted_df, test_dataframe, error_df)

	@classmethod
	def run(cls, dataset_file, k_fold, k_neighbors, k_iterations):
		validator = cls(k_fold, k_neighbors, k_iterations)
		validator.load_data(dataset_file)
		for validation in validator.validate():
			if PRINT:
				print('NEW FOLD')
				for index in range(len(validation.predicted)):
					print('{}\t{}\t{}'.format(validation.predicted[index][0],
											  validation.actual.iloc[index][GeoModel.Columns.ALTITUDE],
											  validation.predicted[index][0] - validation.actual.iloc[index][GeoModel.Columns.ALTITUDE]))
				print('Mean error: {}'.format(validation.error))
			yield validation

def run(dataset_file, k_fold, k_neighbors, k_iterations):
	for _ in GeoModelValidator.run(dataset_file, k_fold, k_neighbors, k_iterations):
		print(_.error)


if __name__ == '__main__':
	run('data.csv', 20, 2, 1)
