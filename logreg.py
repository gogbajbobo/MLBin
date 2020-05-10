import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score


def train_logreg_model(pixel_values, neighbor_average, pixel_bin_values):
    pv = pixel_values.flatten()
    na = neighbor_average.flatten()
    pbv = pixel_bin_values.flatten()
    x_train = np.stack([pv, na], axis=-1)
    y_train = np.asarray(pbv)
    train_scaler = preprocessing.StandardScaler()
    train_scaler.fit(x_train)
    train_scaler.transform(x_train)
    logreg = LogisticRegression(C=1, solver='lbfgs').fit(x_train, y_train)
    print(f'mean: {train_scaler.mean_}, var: {train_scaler.var_}, samples_seen: {train_scaler.n_samples_seen_}')
    return logreg, x_train, y_train


def logreg_predict(lr, x, y):
    y_predict = lr.predict(x)
    print(f'LR.coef_ {lr.coef_}, LR.intercept_ {lr.intercept_}, LR.classes_ {lr.classes_}')
    print(f'classification_report {classification_report(y, y_predict)}')
    print(f'prediction score: {lr.score(x, y)}')
    print(f'jaccard score: {jaccard_score(y_predict, y)}\n')
