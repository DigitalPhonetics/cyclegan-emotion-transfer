import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
import pandas as pd

#=========================================================================
#
#  Simple SVM classifier for training, test and evaluation
#  
#  - Arguments: Training and test data
#  - Output: SVM model, prediction, recall, confusion matrix             
#
#=========================================================================


class SVM:
    
    def __init__(self, C, gamma='scale'):
        self.model = svm.SVC(C=C, 
                             gamma=gamma, 
                             kernel='rbf',
                             class_weight='balanced',
                             decision_function_shape='ovr')


    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    
    def predict(self, X_test):
        return self.model.predict(X_test)


    def get_recall(self, y_pred, y_true):
        recall = recall_score(y_true, y_pred, average='macro')
        return recall
    
    
    def get_confusion_matrix(self, y_pred, y_true):
        cm = confusion_matrix(y_true, y_pred)
        return cm
    
    
    def get_support_vectors(self):
        idx = self.model.support_ 
        sv = self.model.support_vectors_
        n_sv = self.model.n_support_
        return idx, sv, n_sv





