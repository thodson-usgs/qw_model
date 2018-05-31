#import statements
import numpy as np
import pandas as pd

from sklearn import linear_model
from datetime import datetime

#from pandas.core.indexes.datetimes import DatetimeIndex
import pandas as pd
class DatetimeIndex(pd.DatetimeIndex):
    """
    """
    #def __init__(self, data=None):
    #    super().__init__(data)

    def year_start(self):
        """ Return the year start for each timestamp in index
        """

        return pd.to_datetime(self.year, format="%Y")

    def year_end(self):
        """ Return the year end for each timestamp in index
        """
        return pd.to_datetime(self.year + 1, format="%Y")

    def decimal_year(self):
        """
        """
        year_part = self - self.year_start()
        year_length = self.year_end() - self.year_start()
        return self.year + year_part.total_seconds()/year_length.total_seconds()


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
        self._constituent = constituent
        self._surrogates = surrogates
        self._datetimeindex = DatetimeIndex(dataframe.index)

    @classmethod
    def from_raw(cls, surrogates, constituents):
        """
        """
        # perform time based merge of constituents and surrogates
        #df = pd.merge_asof
        #model = cls(dataframe, constituent, surrogates)
        #return model
        pass

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
        self.halfwidth_vector = self._halfwidth_vector()

        self.__cached_design_matrix = None
        self.__cached_reg = None #consider making this an array

        # create design matrix?

    def _halfwidth_vector(self):
        """
        """

        halfwidth_vector = np.empty(len(self._surrogates) + 3)
        halfwidth_vector[:-3] = self.surrogate_halfwidth
        halfwidth_vector[:-1] = self.trend_halfwidth
        halfwidth_vector[:-2] = self.seasonal_halfwidth
        halfwidth_vector[:-3] = 0

    def _get_design_matrix(self):
        """
        """
        # check for cached design matrix
        if self.__cached_design_matrix:
            design_matrix = self.__cached_design_matrix

        # otherwise, initialize a new design matrix
        else:
            #import pdb; pdb.set_trace()
            surrogate_values = self._dataframe[self._constituent].values
            decimal_date = self._datetimeindex.decimal_year()

            # allocate matrix with a column for each surrogate and the three time
            # variables: t, cos(t), sin(t)
            design_matrix = np.empty([surrogate_values.shape[0],
                                      len(self._surrogates) + 3])

            design_matrix[:,len(self._surrogates)] = np.log(surrogate_values)
            design_matrix[:,-1] = decimal_date
            design_matrix[:,-2] = np.cos(2*np.pi*decimal_date)
            design_matrix[:,-3] = np.sin(2*np.pi*decimal_date)
            #XXX check sin and cos functions

            self.__cached_design_matrix = design_matrix

        return design_matrix
    
    @staticmethod
    def _design_matrix_distance(design_matrix, design_vector):
        """

        Parametres
        ==========
        design_matrix : array
        design_vector : array
        """
        return np.abs(design_matrix - design_vector)

    def _distance_to_design_matrix(design_vector):
        """
        Paramaters
        ==========
        design_vector : array
            an array with same columns as the design matrix
        """
        if self.__cached_design_matrix:
            design_matrix = self.__cached_design_matrix

        else:
            design_matrix = self._get_design_matrix()

        # XXX


    def _get_response_vector(self):
        """
        """
        constituent_values = self._dataframe[self._constituent].values
        return np.log(constituent_values)

    @staticmethod
    def distance(array_1, array_2):
        """
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
        weights = (1 - (distance/halfwidth_window)**3)**3

        weights =  np.where(weights < halfwidth_window, weights, 0)

        return np.prod(weights, axis=0)

def _fit(self, design_vector, sample_weight=None, method=None):
        """
        Parameters
        ----------
        t : datetime in decimal years
        s : surrogates
        sample_weights : array
            Optionally used to specify sample weights for regression
        """
        #these should access data rather than generate
        #move the generation to init
        M = self._get_design_matrix()
        y = self._get_response_vector()
        x = design_vector

        if not sample_weight:

            sample_distance = distance(M,x)
            sample_weight = self.tricubic_weight(sample_distance,
                                                  self.halfwidth_vector)

        reg = linear_model.LinearRegression(n_jobs=-1)
        reg.fit(M, y, sample_weight = sample_weight)


        self.__model = reg
        # create vector containing dependent variable

        # calculate weights

def _predict_one(self, design_vector):
    """
    """
    model = self._fit(design_vector)
    prediction = model.predict(design_vector)
    return prediction

def predict(self, design_matrix):
    """
    XXX still need to correct for smearing
    """
    prediction = np.zeros(desing_matrix.shape[0])

    for i in design_matrix:
        prediction[i] = self._predict_one(design_matrix[i])


    return prediction


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    def predict(self, surrogates):
        """ Predict a time series, this should be parallized

        Takes a dataframe, converts surrogates to design matrix. For each row 
        fit a regression, make a prediction, and return all predictions
        Parameters
        ----------

        """
        pass


    def _predict(self, t, s):
        """ Predict time series given regression

        inputs
        """
        #convert surrogates to design matrix

        return self.__model.predict(design_matrix)


    def resid(self):
        """
        """
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
        """
        # check for cached results
        if self.__loo:
            return self.__loo

        else:
            pass
