from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_selection.relevance import calculate_relevance_table
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class FeatureExtractor:
    def __init__(self):
        pass

    def fit(self, X_train, y):
        # Reshaping the dataframe to fit tsfresh requirements :
        columns = [
            "DoseRate",
            "SimulatedGamma",
            "Precipitation",
            "InternalTemperature",
            "Pressure",
            "Temperature",
            "ZonalWind",
        ]
        X_tsfresh = pd.DataFrame([], columns=columns)
        X_tsfresh["Key"] = np.concatenate(
            [np.array([k] * X_train.at[k, "Event_length"]) for k in X_train.index]
        )
        for i, data in enumerate(columns):
            X_tsfresh[data] = np.concatenate(
                [
                    X_train.loc[
                        X_train.index == k,
                        list(
                            range(
                                X_train.at[k, "Event_length"] * i,
                                X_train.at[k, "Event_length"] * (i + 1),
                            )
                        ),
                    ].values
                    for k in X_train.index
                ],
                axis=1,
            ).reshape([X_tsfresh.shape[0], 1])
        X_tsfresh = X_tsfresh.drop(columns="InternalTemperature").astype(
            "float64"
        )  # Drop deficient column and force column type

        extracted_features = extract_features(X_tsfresh, column_id="Key")
        # Imputation to correct values equal to inf or nan
        extracted_features = impute(extracted_features)
        # Calculation of p_values to evaluate features usefullness
        df_p_values = calculate_relevance_table(
            extracted_features, y, ml_task="classification"
        )
        self.df_p_value = df_p_values
        # Deleting irrelevant features according to p_values
        extracted_features = extracted_features.drop(
            columns=df_p_values[df_p_values.relevant == 0].index.values
        )
        # Dictionnary containing features and corresponding parameters
        self.features = extracted_features.columns

    def transform(self, X):
        # Reshaping the dataframe to fit tsfresh requirements :
        columns = [
            "DoseRate",
            "SimulatedGamma",
            "Precipitation",
            "InternalTemperature",
            "Pressure",
            "Temperature",
            "ZonalWind",
        ]
        X_tsfresh = pd.DataFrame([], columns=columns)
        X_tsfresh["Key"] = np.concatenate(
            [np.array([k] * X.at[k, "Event_length"]) for k in X.index]
        )
        for i, data in enumerate(columns):
            X_tsfresh[data] = np.concatenate(
                [
                    X.loc[
                        X.index == k,
                        list(
                            range(
                                X.at[k, "Event_length"] * i,
                                X.at[k, "Event_length"] * (i + 1),
                            )
                        ),
                    ].values
                    for k in X.index
                ],
                axis=1,
            ).reshape([X_tsfresh.shape[0], 1])
        X_tsfresh = X_tsfresh.drop(columns="InternalTemperature").astype(
            "float64"
        )  # Drop deficient column and force column type

        X = extract_features(X_tsfresh, column_id="Key")
        return impute(X[self.features])
