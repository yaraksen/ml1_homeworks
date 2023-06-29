from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from scipy.special import softmax


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        mask = np.random.choice(x.shape[0], int(self.subsample * x.shape[0]), replace=True)
        s = -self.loss_derivative(y[mask], predictions[mask])
        clf = self.base_model_class(**self.base_model_params)
        clf.fit(x[mask], s)

        gammas = self.find_optimal_gamma(y[mask], predictions[mask], clf.predict(x[mask]))

        self.gammas.append(gammas)
        self.models.append(clf)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        best_score, no_better_score_counter = 0, 0
        for i in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            train_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_train)
            self.history['train'].append(self.loss_fn(y_train, train_predictions))

            valid_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid)
            # self.validation_loss[i] = self.loss_fn(y_valid, valid_predictions)
            self.history['validation'].append(self.loss_fn(y_train, train_predictions))

            if self.early_stopping_rounds is not None:
                if no_better_score_counter == self.early_stopping_rounds:
                    break
                new_score = self.score(x_valid, y_valid)
                if new_score < best_score:
                    no_better_score_counter += 1
                else:
                    best_score = new_score
                    no_better_score_counter = 0

        if self.plot:
            sns.lineplot(data=self.history)

    def predict_proba(self, x):
        predictions = np.zeros(x.shape[0])
        for model, gamma in zip(self.models, self.gammas):
            predictions += self.learning_rate * gamma * model.predict(x)
        probas = self.sigmoid(predictions).reshape(-1, 1)
        return np.hstack((1 - probas, probas))

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        feature_importances_sum = np.array(self.models[0].feature_importances_)
        for i in range(1, len(self.models)):
            feature_importances_sum += np.array(self.models[i].feature_importances_)
        feature_importances_sum /= len(self.models)
        return feature_importances_sum / np.sum(feature_importances_sum)
