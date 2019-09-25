import numpy as np
import pandas as pd
import os
import xarray

#=========================================================================
#
#  This file is used to load data from netCDF (.nc) files.
#  
#  - Arguments: path to the .nc file
#  - Output: feature and label data               
#
#=========================================================================


class InputData:


    def __init__(self, path):
        self.data = xarray.open_dataset(path)
        self.file_name = self.data.file_name.to_pandas()
        self.feature_name = self.data.feature_name.to_pandas()    
        self.feature_value = self.data.feature_value.to_pandas()
        if "category" in self.data.variables:
            self.category = self.data.category.to_pandas()
        if "arousal" in self.data.variables:
            self.arousal = self.data.arousal.to_pandas()
        if "valence" in self.data.variables:
            self.valence = self.data.valence.to_pandas()

        
    def get_data(self):
        """
        Get an object of xarray.Dataset
        """
        return self.data
    
    
    def get_session(self, session_prefix, output=True):
        """
        Get the names of all files which belong to the given session
        @param session_prefix: Prefix that files of one session have
                               in common with, e.g. All the files in 
                               session 1 have prefix "Ses01"
        @return session: List of file names in this session
        """
        session = [name for name in self.file_name \
                if name.startswith(session_prefix)]
        if output:
            print("{} size: {}".format(session_prefix, 
                np.array(session).shape))
        return session
    
    
    def _extract(self, df, indices=None):
        """
        Extract the items at the given indices from the dataframe.
        If indices not given, all items in the dataframe will be returned
        @param df: Dataframe to be extracted from
        @param indices: Indices of the items to be extracted
        @return df: Dataframe only having the items at the indices
        """
        if indices:
            return df.loc[df.index.isin(indices)]
        return df
    
    
    def get_features(self, indices=None):
        """
        Get feature values of the items at the given indices 
        """
        return self._extract(self.feature_value, indices)
    

    def get_category(self, indices=None):
        """
        Get category of the items at the given indices
        """
        return self._extract(self.category, indices)


    def get_arousal(self, indices=None):
        """
        Get arousal of the items at the given indices 
        """
        return self._extract(self.arousal, indices)
    
 
    def get_valence(self, indices=None):
        """
        Get valence of the items at the given indices 
        """
        return self._extract(self.valence, indices)
    