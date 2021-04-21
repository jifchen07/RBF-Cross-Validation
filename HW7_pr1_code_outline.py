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
import matplotlib.pyplot as plt

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
    # M is scalar
    if mode == 1:  # (c)
        centers_ = X
    elif mode == 2:  # (d)
        np.random.seed(0)
        random_idxs_ = np.random.choice(X.shape[0], size=M, replace=False)
        centers_ = X[random_idxs_, :]
    elif mode == 3:  # (e)
        # K-means clustering
        kmeans = KMeans(n_clusters=M, init='random', random_state=0).fit(X)
        centers_ = kmeans.cluster_centers_

    return centers_


def cal_mse(y, y_pred):
    """Calculate MSE"""
    mse = mean_squared_error(y, y_pred)
    return mse


# def calculate_gamma(X, M):
#     """Calculate gamma"""
#     avg_spacing_ = cal_avg_spacing(X, M)
#     sigma_ = 2 * avg_spacing_
#     gamma_ = 1 / (2 * sigma_**2)
#     return gamma_
#
#
# def cal_avg_spacing(X, M):
#     """Calculate average spacing"""
#     n_features = X.shape[1]
#     delta_acc = 1
#     for i in range(n_features):
#         delta = np.max(X[:, i]) - np.min(X[:, i])
#         delta_acc *= delta
#     avg_space = (delta_acc / M) ** (1 / n_features)
#     return avg_space

def calculate_gamma_all(X, M):
    # M is 1d array
    n_features = X.shape[1]
    delta_acc = 1
    for i in range(n_features):
        delta = np.max(X[:, i]) - np.min(X[:, i])
        delta_acc *= delta
    gamma_ = 1 / (8 * delta_acc) * M
    return gamma_

def visualize(X, y):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, marker='o')
    plt.show()

if __name__ == "__main__":
    '''Example structure'''
    MODE = 2  # Example: mode == 3: using K-means for center selection
    v = np.linspace(0.02, 0.2, num=10)
    K = 24  # value got from mode 2

    # ------------- Load data -------------
    X, y = load_data('H7_Dn_train.csv')
    # visualize(X, y)

    # (b) mse for trivial system
    ytr_mean = np.mean(y)
    mse_trivial = cal_mse(y, np.full(y.shape, ytr_mean))
    print(f'part (b): mse of the trivial system is {mse_trivial}')

    # ============================= Start of "RBFN Module" =============================
    # ------------- Calculate M using v or K -------------

    # # ------------- Select centers -------------
    # centers = choose_centers(X, M, mode=MODE)
    #
    # # ------------- Start 2 layers of RBF Network -------------
    # # Layer 1
    # phi_x = rbf_layer1(X, centers, gamma)   # (n_samples, M)
    # # Layer 2
    # linear_layer2 = rbf_layer2(phi_x, y)
    # y_hat = rbf_layer2(phi_x, weight=linear_layer2)
    #
    # # ------------- Calculate MSE -------------
    # MSE = cal_mse(y, y_hat)
    #
    # print(f'mse of training: {MSE}')

    # for cross validation

    # ------------- Find average spacing and initial gamma -------------
    # avg_spacing = cal_avg_spacing(X, M)
    # M <-> gamma <-> centers

    T = 20
    n_splits = 5

    training_size = X.shape[0] * (n_splits - 1) // n_splits

    if MODE == 1:
        M = np.array([training_size])
    elif MODE == 2:
        M = (v * training_size).astype(int)
    elif MODE == 3:
        M = np.linspace(K-12, K+12, num=7, dtype=int)

    gammas = calculate_gamma_all(X, M)   # (n_M, )   gamma baselines
    variations = np.logspace(-2, 2, num=5, base=2)  # (n_M, n_g)    gamma variations

    mses = np.zeros((len(gammas), len(variations), T, n_splits))

    for t in range(T):
        print(f'cross validation run {t+1}')
        skf = KFold(n_splits=n_splits, random_state=t, shuffle=True)
        for k, (tr_idxs, va_idxs) in enumerate(skf.split(X, y)):
            for i, gamma in enumerate(gammas):
                centers = choose_centers(X[tr_idxs], M[i], mode=MODE)   # each gamma/M will generate different center
                for j, var in enumerate(variations):
                    gamma_var = gamma * var # apply variation to gamma
                    phi_X = rbf_layer1(X, centers, gamma_var)
                    phi_X_tr, phi_X_va = phi_X[tr_idxs], phi_X[va_idxs]
                    y_tr, y_va = y[tr_idxs], y[va_idxs]

                    linear_layer2 = rbf_layer2(phi_X_tr, y_tr)
                    y_hat = rbf_layer2(phi_X_va, weight=linear_layer2)

                    # if i == 2 and j == 0:
                    #     print(gamma, var)
                    #     visualize(X[va_idxs], y_hat)
                    mses[i][j][t][k] = cal_mse(y_va, y_hat)

    mses = mses.reshape((len(gammas), len(variations), -1))
    mse_means = np.mean(mses, axis=2)
    mse_stds = np.std(mses, axis=2)

    min_idx = np.unravel_index(mse_means.argmin(), mse_means.shape)
    gamma_opt = gammas[min_idx[0]] * variations[min_idx[1]]
    M_opt = M[min_idx[0]]
    std_opt = mse_stds[min_idx]
    print(f'when gamma is {gamma_opt:.3}, M is {M_opt}, the average validation error is the smallest:\n'
          f'mean is {mse_means[min_idx]:.3}, std is {mse_stds[min_idx]:.3}')
    # ============================= End of "RBFN Module" =============================

    # mses = [mse / (T * n_splits) for mse in mses]
    print('finished!')