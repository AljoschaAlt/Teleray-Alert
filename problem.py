import os
import pandas as pd
import rampwf as rw
import pickle
from rampwf.score_types import BaseScoreType
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

problem_title = 'Teleray alert: Radiological events classification'

#lalel names for the classification
_prediction_label_names = ['Weather','AtmosphericRejection','ElectronicPeak1', 'ElectronicPeak2', 'ElectronicPeak3']

Predictions = rw.prediction_types.make_multiclass(
label_names=_prediction_label_names)

#Workflow
workflow = rw.workflows.FeatureExtractorClassifier()

class weighted_ROCAUC(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='weighted_roc_auc', precision=2):
        self.name = name
        self.precision = precision

    def score_function(self, ground_truths, predictions):
        """
            Weighted average AUC:
            Calculate metrics for each label, and find their average, weighted by support. 
        """
        print('predictions')
        print(predictions)
        print('-')
        print(Predictions.label_names)
        print('label')
        print(Predictions.y_pred_label)
        print('index')
        print(Predictions.y_pred_label_index)
        y_proba = predictions.y_pred #shape (n_samples, n_classes)
        y_true = ground_truths.y_pred_label_index #shape (n_samples, 1)
        y_true= label_binarize(y_true, classes=np.unique(y_true)) #shape (n_samples, n_classes)
        self.check_y_pred_dimensions(y_true, y_proba)
        print('----------')
        return self.__call__(y_true, y_proba)

    def __call__(self, y_true, y_proba):
        return roc_auc_score(y_true, y_proba, average='weighted')


#Scores
score_types = [
    weighted_ROCAUC(),
    rw.score_types.BalancedAccuracy(name='acc')]

#Cross-validation scheme
def get_cv(X, y):
    cv = StratifiedKFold(n_splits=4, random_state=2, shuffle=True)
    return cv.split(X, y)

#Data reading 
def _read_data(f_path):
    X= pickle.load(open(f_path, 'rb'))
    y= X.Label
    return X, y

def get_train_data(path='.'):
    f_name= 'train.pickle'
    f_path=os.path.join(path,'data/public',f_name)
    return _read_data(f_path)


def get_test_data(path='.'):
    f_name='test.pickle'    
    f_path=os.path.join(path,'data/public',f_name)
    return _read_data(f_path)
