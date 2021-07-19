{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "13a53769",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import rampwf as rw\n",
    "import pickle\n",
    "from rampwf.score_types import BaseScoreType\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.metrics import roc_auc_score\n",
    "from sklearn.preprocessing import label_binarize"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "01d0bb05",
   "metadata": {},
   "outputs": [],
   "source": [
    "problem_title = 'Teleray alert: Radiological events classification'\n",
    "\n",
    "#lalel names for the classification\n",
    "_prediction_label_names = ['Weather','AtmosphericRejection','ElectronicPeak1', 'ElectronicPeak2', 'ElectronicPeak3']\n",
    "\n",
    "Predictions = rw.prediction_types.make_multiclass(\n",
    "label_names=_prediction_label_names)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9198fc4e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Workflow\n",
    "workflow = rw.workflows.FeatureExtractorClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "f11f81c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "class weighted_ROCAUC(BaseScoreType):\n",
    "    is_lower_the_better = False\n",
    "    minimum = 0.0\n",
    "    maximum = 1.0\n",
    "\n",
    "    def __init__(self, name='weighted_roc_auc', precision=2):\n",
    "        self.name = name\n",
    "        self.precision = precision\n",
    "\n",
    "    def score_function(self, ground_truths, predictions):\n",
    "        \"\"\"\n",
    "            Weighted average AUC:\n",
    "            Calculate metrics for each label, and find their average, weighted by support. \n",
    "        \"\"\"\n",
    "        y_proba = predictions.y_pred #shape (n_samples, n_classes)\n",
    "        y_true_proba = ground_truths.y_pred_label_index #shape (n_samples, 1)\n",
    "        y_true_proba= label_binarize(y_true_proba, classes=np.unique(y_true_proba)) #shape (n_samples, n_classes)\n",
    "        self.check_y_pred_dimensions(y_true_proba, y_proba)\n",
    "        return self.__call__(y_true_proba, y_proba)\n",
    "\n",
    "    def __call__(self, y_true_proba, y_proba):\n",
    "        return roc_auc_score(y_true_proba, y_proba, average='weigthed')\n",
    "\n",
    "\n",
    "#Scores\n",
    "score_types = [\n",
    "    weighted_ROCAUC(),\n",
    "    rw.score_types.BalancedAccuracy(name='acc')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f944b8d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Cross-validation scheme\n",
    "def get_cv(X, y):\n",
    "    cv = StratifiedKFold(n_splits=5, random_state=2)\n",
    "    return cv.split(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4f4786f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Data reading \n",
    "def _read_data(f_name_events, f_name_timeSeries):\n",
    "    df_events= pickle.load(open(f_name_events, 'rb'))\n",
    "    df_timeSeries= pickle.load(open(f_name_timeSeries, 'rb'))\n",
    "    y= df_events.Label\n",
    "    return df_events, df_timeSeries, y\n",
    "\n",
    "def get_train_data(path='.'):\n",
    "    f_name_events = 'train_events.pickle'\n",
    "    f_name_timeSeries = 'train_time_series.pickle'\n",
    "    \n",
    "    f_name_events=os.path.join(path,'data/public',f_name_events)\n",
    "    f_name_timeSeries=os.path.join(path,'data/public',f_name_timeSeries)\n",
    "    return _read_data(f_name_events, f_name_timeSeries)\n",
    "\n",
    "\n",
    "def get_test_data():\n",
    "    f_name_events = 'test_events_ramp.pickle'\n",
    "    f_name_timeSeries = 'test_time_series.pickle'\n",
    "    \n",
    "    f_name_events=os.path.join(path,'data/public',f_name_events)\n",
    "    f_name_timeSeries=os.path.join(path,'data/public',f_name_timeSeries)\n",
    "    return _read_data(f_name_events, f_name_timeSeries)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
