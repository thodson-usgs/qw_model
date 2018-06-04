#import statements
import numpy as np
import pandas as pd

from sklearn import linear_model
from datetime import datetime

from qw_model import TimeSeriesDataFrame, TimeSeriesSurrogateModel

class WRSSTCache():
    """ Cache for WRSSTDataFrame

    Stores commonly accessed variables, so that they don't have to be regenerated.
    """
    def __init__(self):
        # initialize array variables
        self.design_matrix = np.array([])
        self.reg = np.array([])
        self.loo = np.array([])#consider making this an array
        self.smearing_coef = np.array([])
        self.predicted_response = np.array([])

class WRSSTDataFrame(TimeSeriesDataFrame):
    """XXX Need to rename this to table
    """
    def __init__(self, dataframe, constituents, surrogates):
        super().__init__(dataframe, constituents, surrogates)
        self._cache = WRSSTCache()

    @property
    def design_matrix(self):
        """
        modify to call _obesrvations_to_design
        """
        # check for cached design matrix
        if self._cache.design_matrix.size > 0:
            design_matrix = self._cache.design_matrix

        # otherwise, initialize a new design matrix
        else:
            surrogate_values = self.surrogates.values
            decimal_date = self._datetimeindex.decimal_year()

            # allocate matrix with a column for each surrogate and the three time
            # variables: t, cos(t), sin(t)
            design_matrix = np.empty([surrogate_values.shape[0],
                                      self.surrogate_count + 3])

            #expand dimensions of surrogate_values if neccessary
            if surrogate_values.ndim == 1:
                surrogate_values = np.expand_dims(surrogate_values, 1)

            design_matrix[:,0:self.surrogate_count] = np.log(surrogate_values)
            design_matrix[:,-1] = decimal_date
            design_matrix[:,-2] = np.cos(2*np.pi*decimal_date)
            design_matrix[:,-3] = np.sin(2*np.pi*decimal_date)
            #XXX check sin and cos functions

            self._cache.design_matrix = design_matrix

        return design_matrix

    @property
    def response_vector(self):
        """
        Return
        ------
        An array of log transformed constituent observatios.
        """
        return np.log(self.constituent.values)

    @property
    def predicted_response(self):
        return self._cache.predicted_response

    @predicted_response.setter
    def predicted_response(self, array):
        self._cache.predicted_response = array

    @property
    def residuals(self):
        """
        """
        return self.response_vector - self.predicted_response


class WRSST(TimeSeriesSurrogateModel):
    """ Weighted Regression on Surrogates, Season, and Trend.

    """

    def __init__(self, dataframe,
                 constituent,
                 surrogates,
                 time_step=1/365.5,
                 surrogate_hw =3.0, #2.0 orig
                 seasonal_hw = 1, #0.5 orig
                 trend_hw = 7.0):
        """ Initialize a WRSST model

        Parameters
        ----------
        dataframe : DataFrame
        time_step : float
        surrogate_hw : float
        seasonal_hw : float
        trend_hw : float
        """
        self.input_data = WRSSTDataFrame(dataframe, constituent, surrogates)
        self.seasonal_halfwidth = seasonal_hw
        self.surrogate_halfwidth = surrogate_hw
        self.trend_halfwidth = trend_hw
        self.time_step = time_step

        self._halfwidth_vector = np.array([])

        # calculate model residuals
        observed = self.input_data.response_vector
        self.predicted_response, self.input_data.smearing_coef = self.predict(self.input_data.design_matrix) #move to cache XXX

        # move this to a utils module, along with distance and weight


    @staticmethod
    def _calculate_duan_smearing_coef(weights, residuals):
        """
        Parameters
        ----------
        weights : array
        residuals : array
        """
        weighted_sum_of_residuals = (weights * np.exp(residuals)).sum()
        sum_of_weights = weights.sum()
        return weighted_sum_of_residuals / sum_of_weights

    @property
    def halfwidth_vector(self):
        """
        """
        if self._halfwidth_vector.size > 0:
            halfwidth_vector = self._halfwidth_vector

        else:
            halfwidth_vector = np.empty(self.input_data.surrogate_count + 3)
            halfwidth_vector[:-3] = self.surrogate_halfwidth
            halfwidth_vector[-1] = self.trend_halfwidth
            halfwidth_vector[-2] = self.seasonal_halfwidth
            halfwidth_vector[-3] = self.seasonal_halfwidth
            self._halfwidth_vector = halfwidth_vector

        return halfwidth_vector


    @staticmethod
    def distance(array_1, array_2):
        """ Calculate distance bt
        """
        return np.abs(array_1 - array_2)

    @staticmethod
    def tricubic_weight(distance, halfwidth_window):
        """ Tricube weight function (Tukey, 1977).

        Parameters
        ----------
        distance : array
        halfwidth_window : array

        Returns
        -------
        An array of weights.
        """
        x = np.divide(distance, halfwidth_window,
                      out=np.zeros_like(distance),
                      where=halfwidth_window!=0)

        weights = (1 - x**3)**3

        weights =  np.where(distance <= halfwidth_window, weights, 0)

        return np.prod(weights, axis=1)

    def _fit(self, design_vector,
             design_matrix=None,
             response_vector=None,
             sample_weight=None,
             method=None):
            """
            Parameters
            ----------
            design_vector : array
                The observation to base the LOWESS off

            mple_weightdesign_matrix : array
                An array of all observations

            sample_weights : array
                Optionally used to specify sample weights for regression

            method : string
                Not implemented
            """
            #these should access data rather than generate
            #move the generation to init
            if design_matrix is None:
                M = self.input_data.design_matrix
            else:
                M = design_matrix

            if response_vector is None:
                y = self.input_data.response_vector
            else:
                y = response_vector

            x = design_vector

            if sample_weight is None:
                self._calculate_sample_weights(M,x)


            reg = linear_model.LinearRegression(n_jobs=-1)
            reg.fit(M, y, sample_weight = sample_weight)

            return reg


            #self.__model = reg
            # create vector containing dependent variable XXX

            # calculate weights

    def _calculate_sample_weights(self, design_vector, design_matrix):
        """
        Parameters
        ==========
        design_vector : array
            Array containing explantory variables at current observations

        design_matrix : array
            Array containing all

        Returns
        =======
        An array containing the weights of each sample relative to the observations
        """
        sample_distance = self.distance(design_matrix, design_vector)
        sample_weight = self.tricubic_weight(sample_distance,
                                            self.halfwidth_vector)

        return sample_weight

    def _predict_one(self, design_vector):
        """
        """
        model = self._fit(design_vector)
        prediction = model.predict(design_vector.reshape(1,-1))
        return prediction

    def predict(self, design_matrix):
        """
        XXX rename design matrix so as not to confuse with training matrix

        Return
        ======
        prediction : array
        smearing_coef : array
        """
        number_of_observations = design_matrix.shape[0]
        prediction = np.zeros(number_of_observations)
        residuals = np.zeros(number_of_observations)
        smearing_coef = np.zeros(number_of_observations)
        weight = np.zeros([number_of_observations, number_of_observations])

        for i, row in enumerate(design_matrix):
            # calculate weights
            weight[i] = self._calculate_sample_weights(row, self.input_data.design_matrix)

            # fit model
            local_regression = self._fit(row, sample_weight=weight[i])

            # make prediction
            prediction[i] = local_regression.predict(row.reshape(1,-1))

            residuals[i] = self.input_data.response_vector[i] - prediction[i]
            smearing_coef[i] = self._calculate_duan_smearing_coef(weight, residuals)

        return prediction, smearing_coef


    def residuals(self):
        """ Calculate model residuals

        """
        return self.cache
        # check for cached results
        if self.__resid:
            return self.__resid

        else:
            #
            pass


        for i in df:
            pass

        return predicted, residuals

    def loo(self):
        """Return leave one out resduals

        For entr
        """
        # check for cached results
        if self.cache.loo.size > 0:
            return self.__loo

        else:
            prediction = np.zeros(self._design_matrix.shape[0])
            #for entriy in design matrix
            for i, row in enumerate(self._design_matrix):
                # make design matrix with one left out
                loo_matrix = np.delete(self._design_matrix, i, 0)
                #make response vector with one left out
                loo_response = np.delete(self._get_response_vector(), i ,0)

                model = self._fit(row, design_matrix=loo_matrix,
                          response_vector=loo_response)

                prediction[i] = model.predict(row.reshape(1,-1))

            return prediction
