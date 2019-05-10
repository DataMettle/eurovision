# From https://github.com/mgorjis/ITCC

import numpy as np


def prob_clust_indiv(p, x_hat, c_x, y_hat, c_y):
    return np.sum(
        p[np.array(c_x == x_hat).ravel(), :][:, np.array(c_y == y_hat).ravel()])


def prob_clust(p, x_hat, c_x, y_hat, c_y):
    output = np.empty([len(x_hat), len(y_hat)], dtype=float)
    for xhat in x_hat:
        for yhat in y_hat:
            output[xhat, yhat] = prob_clust_indiv(p, xhat, c_x, yhat, c_y)
    return output


def prob_x_given_xhat(p, x, xhat, c_x):
    return np.sum(p[x, :]) / np.sum(p[np.array(c_x == xhat).ravel(), :])  #


def prob_y_given_yhat(p, y, yhat, c_y):
    return np.sum(p[:, y]) / np.sum(p[:, np.array(c_y == yhat).ravel()])  #


def prob_y_given_x(p, x):
    return p[x, :] / np.sum(p[x, :])


def prob_x_given_y(p, y):
    return p[:, y] / np.sum(p[:, y])


def calc_q_indiv(p, x, c_x, y, c_y):
    return prob_clust_indiv(p, c_x[0, x], c_x, c_y[0, y], c_y) * prob_x_given_xhat(
        p, x, c_x[0, x], c_x) * prob_y_given_yhat(p, y, c_y[0, y], c_y)


def calc_q(p, x, c_x, y, c_y):
    output = np.empty([len(x), len(y)], dtype=float)
    for xx in x:
        for yy in y:
            output[xx, yy] = calc_q_indiv(p, xx, c_x, yy, c_y)
    return output


def prob_y_given_xhat(p, x_hat, c_x):
    return np.sum(p[np.array(c_x == x_hat).ravel(), :] / np.sum(
        p[np.array(c_x == x_hat).ravel(), :]), axis=0)


def prob_x_given_yhat(p, y_hat, c_y):
    return np.sum(p[:, np.array(c_y == y_hat).ravel()] / np.sum(
        p[:, np.array(c_y == y_hat).ravel()]), axis=1)


def kl_divergence(p, q):
    tolerance = 0.00000000000000000001
    # Big=1000000
    p = np.asmatrix(p, dtype=np.float)
    q = np.asmatrix(q, dtype=np.float)
    s = 0
    m = np.shape(p)[0]
    n = np.shape(p)[1]
    for i in range(0, m):
        for j in range(0, n):
            kl = (p[i, j] + tolerance) / (q[i, j] + tolerance)  # +TOLERANCE
            s = s + (p[i, j] * np.log2(kl))
    return s


def next_cx(p, q, x, c_x, k):
    q_dist_xhat = np.empty(k)
    p_dist_x = prob_y_given_x(p, x)

    for xhat in range(0, k):
        q_dist_xhat[xhat] = kl_divergence(
            p_dist_x.ravel(),
            prob_y_given_xhat(q, xhat, c_x).ravel()
        )
        # print(q_dist_xhat)
    return np.argmin(q_dist_xhat)  #


def next_c_x(p, q, c_x, k):
    output = np.empty(np.shape(c_x)[1])
    for x in range(0, np.shape(c_x)[1]):
        # print(x)
        output[x] = next_cx(p, q, x, c_x, k)
    return output


def next_cy(p, q, y, c_y, l):
    q_dist_yhat = np.empty(l)
    p_dist_y = prob_x_given_y(p, y)
    for yhat in range(0, l):
        q_dist_yhat[yhat] = kl_divergence(
            p_dist_y.ravel(),
            prob_x_given_yhat(q, yhat, c_y).ravel()
        )
    return np.argmin(q_dist_yhat)


def next_c_y(p, q, c_y, l):
    output = np.empty(np.shape(c_y)[1])
    for y in range(0, np.shape(c_y)[1]):
        output[y] = next_cy(p, q, y, c_y, l)
    return output


def sorting(p, k, l, c_x, c_y):
    n = np.shape(p)[1]
    m = np.empty((1, n))
    m = np.delete(m, 0, axis=0)

    for i in range(0, k):
        indexes = np.where(c_x == i)
        a = p[indexes[1], :]
        m = np.vstack([m, a])

    m1 = np.empty((m, 1))
    m1 = np.delete(m1, 0, axis=1)

    for j in range(0, l):
        indexes = np.where(c_y == j)
        a = m[:, indexes[1]]
        m1 = np.hstack([m1, a])

    return m1
