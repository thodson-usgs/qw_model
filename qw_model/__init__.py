#import statements
import numpy as np
import pandas as pd

from sklearn import linear_model
from datetime import datetime

from pandas.core.indexes.datetimes import DatetimeIndex

class DatetimeIndex(DatetimeIndex):
    """
    """
    def __init__(self):
        super().__init__()

    def year_start(self):
        """ Return the year start for each timestamp in index
        """

        return pd.to_datetime(self.year, format="%Y")

    def year_end(self):
        """ Return the year end for each timestamp in index
        """
        return pd.to_datetime(self.year + 1, format="%Y")

    def decimalyear(self):
        """
        """
        year_part = ts - year_start(ts)
        year_length = year_end(ts) - year_start(ts)
        return dt.year + year_part/year_length


class HierarchicalSurrogateModel:
    pass

class TimeSeriesSurrogateModel:
    """ A time-series surrogate model
    """
    def __init__(self, dataframe,
                constituent,
                surrogates):
        """
        """
        self._dataframe = dataframe
        self._constutent = constituent
        self._surrogates = surrogates
        self._datetimeindex = DatetimeIndex(dataframe.index)


    def _constituent(self, array=True):
        """
        """
        if array:
            return dataframe[self._constituent].values

        else:
            return dataframe[self._constituent]


    def _surrogate(self, array=True):
        """
        """
        if array:
            return self.dataframe[self._surrogates].values

        else:
            return dataframe[self._surrogate]

    def _timeseries(self, array=True):
        """
        """
        pass


class WRSST(TimeSeriesSurrogateModel):
    """
    """

    def __init__(self, dataframe,
                 constituent,
                 surrogates,
                 time_step=1/365.5,
                 surrogate_hw =2.0,
                 seasonal_hw = 0.5,
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
        super().__init__(dataframe, constituent, surrogates)
        self.seasonal_halfwidth = seasonal_hw
        self.surrogate_halfwidth = surrogate_hw
        self.trend_halfwidth = trend_hw
        self.time_step = time_step

        self.__cached_design_matrix = None
        self.__cached_reg = None #consider making this an array

    def predict(self, method='ols'):
        """ Predict time series given regression
        """

        pass

    def fit(self, t, s, sample_weights=None, method=None):
        """
        Parameters
        ----------
        t : datetime in decimal years
        sample_weights : array
            Optionally used to specify sample weights for regression
        """
        #these should access data rather than generate
        M = self.__design_matrix()
        y = np.log(self.__constituent_array())

        if not sample_weights:
            sample_weights = self.time_weight(t) * self.surrogate_weight(s)

        if method == 'ols':
            # use ordinary least squares
            reg = linear_model.LinearRegression(n_jobs=-1)

            reg.fit(M, y, sample_weight = sample_weights)

        reg.sample_weights = sample_weights

        return reg



        # create vector containing dependent variable

        # calculate weights

        # save all parameters and all r2


    def __design_matrix(self):
        """ Create the design matrix.

        For example to fit a quadratic polynomial of the form y= a + b*x, the
        create a design matrix with constant column of 1s and a column
        containing x.
        """
        #modify to use a cached design matrix
        return 0


    def surrogate_weight(self, surrogate_obs):
        """ Calculate weights for surrogate observations

        Calculate weights of discharge observations using a tricube weight
        function (Tukey, 1977).

        Parameters
        ----------

        Return
        ------
        An array of surrogate weights
        """
        distance = WRSST.surrogate_distance(surrogate_obs, self.surrogates)
        weight = WRSST.tricubic_weight(distance, self.surrogate_halfwidth)

        # return the product of all the surrogate weights
        return np.prod(weight, axis=0)


    def time_weight(self, time_o, annual=False):
        """
        Parameters
        ----------
        time_o : float
            Decimal year of observation which weights are relative to.

        timeseries : array
            An array containing the time of each observation in decimal years.

        Returns
        -------
        An array of time weights for each observations.
        """
        if annual:
            trend_distance = 1

        else:
            trend_distance = self.trend_distance(time_o, timeseries)

        seasonal_distance = self.seasonal_distance(time_o, timeseries)
        seasonal_weight = self.tricubic_weight(seasonal_distance,
                                               self.trend_halfwidth)
        trend_weight = self.tricubic_weight(trend_distance,
                                            self.seasonal_halfwidth)

        # return the product of all time weights
        return seasonal_weight * trend_weight


    @staticmethod
    def seasonal_distance(time_o, time_i):
        """ Compute the seasonal distance between observations

        XXX does this need an abs?
        """
        t_d = WRSST.trend_distance(time_o, time_i)

        return = np.minimum(np.ceil(d_t) - d_t,
                            d_t - np.floor(d_t))


    @staticmethod
    def trend_distance(time_o, time_i):
        """ Compute the trend distance between observations
        """
        return np.abs(time_o - self.time_i)


    @staticmethod
    def surrogate_distance(surrogate_o, surrogate_i):
        """ Compute the distance between surrogate observations
        """
        distance = np.log(surrogate_o) - np.log(surrogate_i)

        return np.abs(distance)


    @staticmethod
    def tricubic_weight(distance, halfwidth_window):
        """ Tricube weight function (Tukey, 1977).

        Parameters
        ----------
        distance : array
        halfwidth_window : float

        Returns
        -------
        An array of weights.
        """
        weights = (1 - (distance/halfwidth_window)**3)**3

        return np.where(weights < halfwidth_window, weights, 0)
