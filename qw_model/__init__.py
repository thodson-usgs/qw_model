#import statements
import numpy as np
import pandas as pd

from sklearn import linear_model
from datetime import datetime


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


class TimeSeriesSurrogateModel:
    """ A time-series surrogate model
    """

    def plot(self):
        pass

class TimeSeriesDataFrame():
    """
    """
    def __init__(self, dataframe,
                constituent,
                surrogates):
        """
        Parameters
        ----------
        dataframe : DataFrame
        constituent : string
            The name of the constituent field
        surrogates : list
            List of surrogate names.
        """
        self._dataframe = dataframe
        self._constituent_name = constituent
        self._surrogate_name  = surrogates
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

    @property
    def surrogate_count(self):
        """
        """
        return len(self._surrogate_name)

    @property
    def constituent(self):
        """
        """
        return self._dataframe[self._constituent_name]

    @property
    def surrogates(self):
        """
        """
        return self._dataframe[self._surrogate_name]

    def timeseries(self):
        """
        """
        pass
