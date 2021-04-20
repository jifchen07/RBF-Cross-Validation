"""
EE559 Spring2021 H7W13 Problem 1 Code Outline (Optional)
** Please note:
(1) Find a good place for regularization (Ridge).
(2) This code outline doesn't contain cross-validation for model selection.
(3) You need to perform cross-validation based on the "RBFN Module", in order to
    do the model selection of gamma*, v*, and K* (and the regularization)
(4) It'll be more convenient for model selection to wrap the "RBFN Module" into one function, or use OOP instead.
(5) See another example in Discussion 11 Week 13.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def load_data(datapath):
    df = pd.read_csv(datapath, header=None)
    x_, y_ = df.values[:, 0:-1], df.values[:, -1]
    return x_, y_


def rbf_layer1(X, mu, gamma):
    """Layer 1: Get transformed phi(x)"""
    # X: (n_samples, n_dim) mu: (M, n_dim) gamma: scalar => phi_x_: (n_samples, M)
    phi_x_ = rbf_kernel(X, mu, gamma)
    return phi_x_


def rbf_layer2(X, y=None, weight=None):
    """Layer 2: fit/inference the regression"""
    # X: (n_samples, M + 1) y: (n_samples, )
    # augmentation
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    if weight is None:
        # train (fit)
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        return model
    else:
        # test (predict)
        model = weight
        y_hat_ = model.predict(X)
        return y_hat_


def choose_centers(X, M, mode=1):
    """Three different ways of selecting centers"""
    if mode == 1:  # (c)
        centers_ = X
    elif mode == 2:  # (d)
        np.random.seed(0)
        random_idxs_ = np.random.choice(X.shape[0], size=M, replace=False)
        centers_ = X[random_idxs_, :]
    elif mode == 3:  # (e)
        # K-means clustering
        centers_ = []

    return centers_


def cal_mse(y, y_pred):
    """Calculate MSE"""
    mse = mean_squared_error(y, y_pred)
    return mse


def calculate_gamma(X, M):
    """Calculate gamma"""
    avg_spacing_ = cal_avg_spacing(X, M)
    sigma_ = 2 * avg_spacing
    gamma_ = 1 / (2 * sigma_**2)
    return gamma_


def cal_avg_spacing(X, M):
    """Calculate average spacing"""
    n_features = X.shape[1]
    delta_acc = 1
    for i in range(n_features):
        delta = np.max(X[:, i]) - np.min(X[:, i])
        delta_acc *= delta
    avg_space = (delta_acc / M) ** (1 / n_features)
    return avg_space


if __name__ == "__main__":
    '''Example structure'''
    MODE = 1  # Example: mode == 3: using K-means for center selection
    v = 0.1
    K = 500

    # ------------- Load data -------------
    X, y = load_data('H7_Dn_train.csv')

    # (b) mse for trivial system
    ytr_mean = np.mean(y)
    mse_trivial = cal_mse(y, np.full(y.shape, ytr_mean))
    print(f'part (b): mse of the trivial system is {mse_trivial}')

    # ============================= Start of "RBFN Module" =============================
    # ------------- Calculate M using v or K -------------
    if MODE == 1:
        M = X.shape[0]
    elif MODE == 2:
        M = int(v * X.shape[0])
    elif MODE == 3:
        M = K

    # ------------- Find average spacing and initial gamma -------------
    avg_spacing = cal_avg_spacing(X, M)
    gamma = calculate_gamma(X, M)
    gammas = gamma * np.logspace(-3, 3, num=7, base=2)

    # ------------- Select centers -------------
    centers = choose_centers(X, M, mode=MODE)

    # ------------- Start 2 layers of RBF Network -------------
    # Layer 1
    phi_x = rbf_layer1(X, centers, gamma)   # (n_samples, M)
    # Layer 2
    linear_layer2 = rbf_layer2(phi_x, y)
    y_hat = rbf_layer2(phi_x, weight=linear_layer2)

    # ------------- Calculate MSE -------------
    MSE = cal_mse(y, y_hat)

    print(f'mse of training: {MSE}')

    # for cross validation


    T = 20
    n_splits = 5

    mses = np.zeros((len(gammas), T, n_splits))
    for t in range(T):
        skf = KFold(n_splits=n_splits, random_state=t, shuffle=True)
        for j, (tr_idxs, va_idxs) in enumerate(skf.split(X, y)):
            for i, gamma in enumerate(gammas):
                phi_X = rbf_layer1(X, centers, gamma)
                phi_X_tr, phi_X_va = phi_X[tr_idxs], phi_X[va_idxs]
                y_tr, y_va = y[tr_idxs], y[va_idxs]

                linear_layer2 = rbf_layer2(phi_X_tr, y_tr)
                y_hat = rbf_layer2(phi_X_va, weight=linear_layer2)

                mses[i][t][j] = cal_mse(y_va, y_hat)

    mses = mses.reshape((len(gammas), -1))
    mse_means = np.mean(mses, axis=1)
    mse_stds = np.std(mses, axis=1)

    min_idx = np.argmin(mse_means)
    print(f'when gamma is {gammas[min_idx]:.3}, the validation error is the smallest:\n'
          f'mean is {mse_means[min_idx]:.3}, std is {mse_stds[min_idx]:.3}')
    # ============================= End of "RBFN Module" =============================

    # mses = [mse / (T * n_splits) for mse in mses]
    print('finished!')