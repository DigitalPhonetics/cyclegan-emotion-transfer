import numpy as np
import pandas as pd
from evaluation.classifiers.ann import ANN
from evaluation.classifiers.svm import SVM

#=========================================================================
#
#  This file is used to evaluate the  performance of classifiers trained
#  only on real, only on synthetic, on a combination of both and on code
#  vectors.             
#
#=========================================================================

class Evaluator:
    
    def __init__(self, X_train, y_train, X_test, y_test, X_syn, y_syn,
                 code_vectors_train=None, code_vectors_test=None,
                 cls=SVM(C=10), repeated=1):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_syn = X_syn
        self.y_syn= y_syn
        self.code_vectors_train = code_vectors_train
        self.code_vectors_test = code_vectors_test
        self.cls = cls
        self.repeated = repeated
                       
    def _evaluate(self, X_train, y_train, X_test, y_test):
        recalls = []
        cms = []
        for _ in range(self.repeated):
            self.cls.fit(X_train, y_train, X_test, y_test)        
            y_pred = self.cls.predict(X_test)
            recall = self.cls.get_recall(y_pred, y_test)
            recalls.append(recall)
            cm = self.cls.get_confusion_matrix(y_pred, y_test)
            cms.append(cm)
        return np.mean(recalls), np.std(recalls), np.mean(cms, axis=0)
    
    def real(self):
        return self._evaluate(self.X_train, self.y_train, 
                              self.X_test, self.y_test)
    
    def code_vectors(self):
        if (self.code_vectors_train is not None and 
            self.code_vectors_test is not None):
            return self._evaluate(self.code_vectors_train, self.y_train, 
                                  self.code_vectors_test, self.y_test)
        return None
    
    def syn(self,):
        return self._evaluate(self.X_syn, self.y_syn, 
                              self.X_test, self.y_test)

    def real_plus_syn(self):
        X_train_all = pd.concat([self.X_train, self.X_syn])
        y_train_all = pd.concat([self.y_train, self.y_syn])
        return self._evaluate(X_train_all, y_train_all, 
                              self.X_test, self.y_test)
    
    