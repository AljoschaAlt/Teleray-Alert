{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "79a1e6c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tsfresh import extract_features\n",
    "from tsfresh.utilities.dataframe_functions import impute\n",
    "from tsfresh.feature_selection.relevance import calculate_relevance_table\n",
    "from tsfresh.feature_extraction.settings import from_columns\n",
    "\n",
    "class FeatureExtractor():\n",
    "    def __init__(self):\n",
    "        pass\n",
    "\n",
    "    def fit(self, X_df, y):\n",
    "        extracted_features = extract_features(X_df, column_id=\"Cle\")\n",
    "        #Imputation to remove unusable features creating nans \n",
    "        extracted_features=impute(extracted_features)  \n",
    "        #Calculation of p_values to evaluate features usefullness\n",
    "        df_p_values=calculate_relevance_table(extracted_features, y, ml_task='classification')\n",
    "        self.df_p_value= df_p_values\n",
    "        #Deleting irrelevant features according to p_values\n",
    "        extracted_features=extracted_features.drop(columns=df_p_values_new[df_p_values_new.relevant==0].index.values)\n",
    "        #Dictionnary containing features and corresponding parameters \n",
    "        self.features=  from_columns(extracted_features.columns)\n",
    "        \n",
    "    def transform(self, X_df):\n",
    "        df_features= extract_features(X_df, column_id=\"Cle\", default_fc_parameters=self.features)\n",
    "        return df_features\n",
    "        "
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
