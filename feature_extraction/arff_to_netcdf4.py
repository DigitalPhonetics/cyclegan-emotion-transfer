import sys
import ast
import numpy as np
import arff
from netCDF4 import Dataset

#=========================================================================
#
#  This file is used to convert an input feature file (.arff) of 
#  opensmile feature data to its netCDF format (.nc).
#  The generated .nc file contains feature values and their 
#  corresponding labels, such as category, arousal or valence, if
#  labels exist.
#  
#  - Arguments: 
#         feature_file: path to the .arff file
#         data_source: Database name, e.g. IEMOCAP or MSP
#  - Output: .nc file in the same directory as input file               
#
#=========================================================================

class Base(object):
    """
    Base class shared by different data sources. Only the function
    get_label() is overridden by subclasses.
    """

    def __init__(self, feature_file, data_source):
        """
        @param feature_file: Path to the .arff file
        @param data_source: Database name, e.g. IEMOCAP
        """
        self.feature_file = feature_file
        self.data_source = data_source
        

    def get_label(self):
        """
        Get label information, overridden by subclasses
        """
        return {}


    def get_features(self):
        """
        Get feature data which includes attributes(feature names), 
        file names and feature values
        @return attributes: List of attribute names, namely all the 
                            attributes in the .arff file except the
                            1st (file name) and the last one (class)
        @return file_name: List of file names (1st attribute) in the 
                           .arff file 
        @return feature_value: List of feature values
        """
        features = arff.load(open(self.feature_file, "rb"))
        # remove the 1st(file name) and last attribute(class)
        # Each attribute "f" is a tuple of attribute name and data type,
        # e.g. (u'pcm_loudness_sma_maxPos', u'NUMERIC')
        attributes = [f[0] for f in features["attributes"][1:-1]]
        data = features["data"]
        feature_value = np.array(data)[:, 1:-1]
        file_name = [f.replace(".wav", "") for f in np.array(data)[:, 0]]
        return attributes, file_name, feature_value
    

    def create_nc_file(self, feature_name, file_name, feature_value,
                      category, arousal=None, valence=None):
        """
        Create netcdf4 file with feature data and label information 
        """
        
        # create a .nc file
        nc = Dataset(self.feature_file.replace(".arff", ".nc"),
                          "w", format="NETCDF4")
        # create dimensions
        file_name_d = nc.createDimension("file_name", len(file_name))
        feature_name_d = nc.createDimension("feature_name", len(feature_name))
        # create variables
        file_name_v = nc.createVariable("file_name", str, ("file_name",))
        feature_name_v = nc.createVariable("feature_name", str, ("feature_name",))
        feature_value_v = nc.createVariable("feature_value", "f4",
                                        ("file_name", "feature_name"))
        if category:
            category_v = nc.createVariable("category", str, ("file_name",))
        if arousal:
            arousal_v = nc.createVariable("arousal", "f4", ("file_name",))
        if valence:
            valence_v = nc.createVariable("valence", "f4", ("file_name",))
        # create attributes
        if category or arousal or valence:
            nc.description = "features and labels of the {} data".format(
                                                            self.data_source)
        else:
            nc.description = "features of the {} data".format(self.data_source)        
        # write data
        feature_value_v[:] = feature_value
        if category:
            category_v[:] = np.array(category)
        if arousal:
            arousal_v[:] = np.array(arousal)
        if valence:
            valence_v[:] = np.array(valence)
        
        file_name_v[:] = np.array(file_name)
        feature_name_v[:] = np.array(feature_name)
        nc.close()


class IEMOCAP(Base):

    def get_label(self):
        """
        Get label values (category, arousal and valence) of IEMOCAP data
        @return {"category": category, "arousal": arousal, 
                 "valence": valence } 
                where
                category looks like  {"Ses01F_impro01_F012": "ang", ...}
                arousal looks like  {"Ses01F_impro01_F012": 3.5, ...}
                valence looks like {"Ses01F_impro01_F012": 2.0, ...}
        """
        PATH = "/projekte/slu/Data/Emotion/IEMOCAP_full_release/file_lists/"
        label_files = ["0-anger.txt", "1-disgust.txt", "2-excitement.txt",
            "3-fear.txt", "4-frustration.txt", "5-happy.txt", 
            "6-neutral.txt", "7-other.txt", "8-sad.txt", 
            "9-surprise.txt", "xxx.txt"]
        category = {}
        arousal = {}
        valence = {}
        for label_file in label_files:
            with open(PATH+label_file, "r") as f:
                for line in f:
                    items = line.split("\t")
                    # "items" is a list, e.g as follows:
                    # [[85.2700 - 88.0200],	Ses01F_impro01_F012, ang, 
                    #  [2.0000, 3.5000, 3.5000]]
                    lst = ast.literal_eval(items[3])
                    category[items[1]] = items[2]
                    arousal[items[1]] = lst[1]
                    valence[items[1]] = lst[0]
        return {"category": category, "arousal": arousal, "valence": valence}
    
    
    def map_features_to_label(self, file_name, labels):
        """
        Find corresponding labels for the feature data
        @param file_name: List of file names in the .arff file 
        @param labels: Labels returned by get_label() 
        @return category_: List of category value in the same order
                           as feature value
        @return arousal_: List of arousal value in the same order as
                          feature value
        @return valence_: List of valence value in the same order as
                          feature value
        """
        def map_to_label(file_name, label, label_name):
            labels_ = []
            # check if the label information exists
            if label:
                for f_name in file_name:
                    # check if the label of this file exists
                    if f_name in label:
                        labels_.append(label[f_name])
                    else:
                        labels_.append("")
                        print("missing {} for {}".format(label_name, f_name))
            return labels_
        category = labels.get("category")
        if category:
            category_ = map_to_label(file_name, category, "category")
        else:
            category_ = None
        arousal = labels.get("arousal")
        if arousal:
            arousal_ = map_to_label(file_name, arousal, "arousal")
        else:
            arousal_ = None
        valence = labels.get("valence")
        if valence:
            valence_ = map_to_label(file_name, valence, "valence")
        else:
            valence_ = None
        return category_, arousal_, valence_


class MSP(Base):

    def get_label(self):
        """
        Get label values (category) of MSP data
        @return {"category": category}
                where
                category looks like  {"Ses01F_impro01_F012": "ang", ...}
        """
        label_file = "/projekte/slu/Data/Emotion/MSP-IMPROV/evaluation_summary.txt"
        category = {}
        emotions = {"anger": "ang", "happiness": "hap", "neutral": "neu", "sadness": "sad"}
        with open(label_file, "r") as f:
            for line in f:
                items = line.split(";")
                file_name = items[0]
                label = items[1]
                if label in emotions:
                    category[file_name] = emotions[label]
        return {"category": category}
    
    
    def filter_samples(self, file_name, feature_value, labels):
        """
        Remove the files with the categories "no_agreement" and "other" from feature data
        """
        file_name_filtered = []
        feature_value_filtered = []
        category_ = []
        category = labels.get("category")
        if category:
            for f in file_name:
                index = file_name.index(f)
                if f in category:
                    file_name_filtered.append(f)
                    feature_value_filtered.append(feature_value[index])
                    category_.append(category[f])
        return file_name_filtered, feature_value_filtered, category_


if len(sys.argv) < 3:
    print("Please give the input feature file(.arff) and data source")
else:
    feature_file = sys.argv[1]
    data_source = sys.argv[2]
    if data_source == "IEMOCAP":
        iemocap = IEMOCAP(feature_file, data_source)
        feature_name, file_name, feature_value = iemocap.get_features()
        labels = iemocap.get_label()
        category, arousal, valence = iemocap.map_features_to_label(
                                     file_name, labels)
        iemocap.create_nc_file(feature_name, file_name, feature_value,
                               category, arousal, valence)
    elif data_source == "MSP":
        msp = MSP(feature_file, data_source)
        feature_name, file_name, feature_value = msp.get_features()
        labels = msp.get_label()
        file_name, feature_value, category = msp.filter_samples(file_name, 
                                                                feature_value, labels)
        msp.create_nc_file(feature_name, file_name, feature_value, category)
