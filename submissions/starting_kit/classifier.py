from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

class Classifier(BaseEstimator):
    def __init__(self):
        self.n_estimators = 100
        self.clf = Pipeline([
            ('normalisation', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=2))
        ])

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)