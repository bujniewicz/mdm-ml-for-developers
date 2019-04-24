"""
K-nearest Neighbors regressor using Jutland 3D Road Map dataset.
"""

from collections import namedtuple
import random

from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as colors
import numpy as np
import pandas as pd

Heatmap = namedtuple('Heatmap', ['predicted', 'actual', 'error'])
Validation = namedtuple('Validation', ['predicted', 'actual', 'error', 'heatmap'])


class MidpointNormalize(colors.Normalize):
    """
    Helper class to make sure that 0 error is white on error plot.
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


class GeoModel():
    """
    Wrapper around k-Nearest Neighbors regressor providing convenience
    methods for working with Jutland 3D Road Map dataset-based dataframes.
    """

    class Columns:
        """
        List of columns and their identifiers in order of appearance in adult dataframe.

        We don't want to use string literals around. It's very error-prone.
        """

        OSM_ID = 'osm_id'
        LATITUDE = 'latitude'
        LONGTITUDE = 'longtitude'
        ALTITUDE = 'altitude'

    def __init__(self, k_neighbors):
        """
        Initialize variables used later on.
        """

        self.k_neighbors = k_neighbors
        self.regressor = None
        self.features = None
        self.values = None

    def set_training_dataframe(self, dataframe):
        """
        Set data points (features) and results (values) according to input dataframe.

        Reset the regressor to ensure it will be retrained before next prediction.
        """

        self.features = dataframe[[self.Columns.LATITUDE, self.Columns.LONGTITUDE]]
        self.values = dataframe[[self.Columns.ALTITUDE]]
        self.regressor = None

    def train(self):
        """
        Train the regressor using current state of feratures and values.
        """

        if self.regressor is None:
            self.regressor = KNeighborsRegressor(self.k_neighbors)
            self.regressor.fit(self.features, self.values)

    def predict(self, features):
        """
        Train the regressor if needed and predict the result for given array of features.
        """

        if self.regressor is None:
            self.train()
        return self.regressor.predict(features)

    
class GeoModelDatasetProcessor():
    """
    Provides method (prep_dataframe) reading from a file in accordance to road map data set
    and returning a dataframe ready to use in a model.
    """

    COLUMNS = [GeoModel.Columns.OSM_ID, GeoModel.Columns.LATITUDE,
               GeoModel.Columns.LONGTITUDE, GeoModel.Columns.ALTITUDE]

    @classmethod
    def prep_dataframe(cls, dataset_file):
        """
        Prepare the dataframe suitable for GeoModel from input dataset file.

        Normalize latitude and longtitude to values within [0, 1].
        """

        dataframe = pd.read_csv(dataset_file, header=None, names=cls.COLUMNS)

        scaler = MinMaxScaler()
        to_normalize = dataframe[[GeoModel.Columns.LATITUDE, GeoModel.Columns.LONGTITUDE]]
        normalized = scaler.fit_transform(to_normalize)

        normalized_dataframe = pd.DataFrame(normalized, columns=[GeoModel.Columns.LATITUDE,
                                                                 GeoModel.Columns.LONGTITUDE])
        normalized_dataframe[GeoModel.Columns.ALTITUDE] = dataframe[GeoModel.Columns.ALTITUDE]
        
        return normalized_dataframe


class GeoModelValidator():
    """
    Provide validataion capabilities for GeoModel.

    See run for entrypoint.
    """

    dataset_processor = GeoModelDatasetProcessor

    def __init__(self, k_fold, k_neighbors, k_iterations):
        """
        Initialize variables used later on.
        """

        self.k_fold = k_fold
        self.k_iterations = k_iterations
        self.model = GeoModel(k_neighbors)
        self.dataframe = None

    @classmethod
    def prep_dataframe(cls, dataset_file):
        """
        Proxy method for dataset_processor's prep_dataframe.
        """

        return cls.dataset_processor.prep_dataframe(dataset_file)

    def validate(self):
        """
        Validate test dataset against model trained with training dataset.
        """

        for _ in range(self.k_iterations):
            predicted, actual, heatmap = self.validate_fold()
            yield Validation(predicted, actual, mean_absolute_error(actual, predicted), heatmap)

    def validate_fold(self):
        """
        Fold the data into random 1/self.k_fold values constituting a test dataframe
        and the rest being training dataframe and perform prediction on resulting trained
        regressor.
        """

        test_rows = random.sample(self.dataframe.index.tolist(), int(len(self.dataframe) / self.k_fold))

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
        """
        Load data and run validation using input dataset and given k-parameters.
        
        This method is a generator that will yield a Validation exactly k_iterations times.
        Each validation uses data folding as defined by k_fold and the validated regressor will use
        k_neighbors to predict.
        """

        validator = cls(k_fold, k_neighbors, k_iterations)
        validator.dataframe = cls.prep_dataframe(dataset_file)
        for validation in validator.validate():
            yield validation

def run(dataset_file, k_fold, k_neighbors, k_iterations):
    """
    Wrapper around GeoModelValidator.run printing mean average error for every iteration
    of validation.
    """

    for _ in GeoModelValidator.run(dataset_file, k_fold, k_neighbors, k_iterations):
        print(_.error)


if __name__ == '__main__':
    run('data.csv', 20, 2, 1)
