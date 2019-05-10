import numpy as np
import src.util as u


def itcc(p, k, l, n_iters, converge_threshold, c_x, c_y, verbose=False):
    """
    From https://github.com/mgorjis/ITCC
    """
    m = np.shape(p)[0]
    n = np.shape(p)[1]

    q = u.calc_q(p, range(0, m), c_x, range(0, n), c_y)
    kl_curr = u.kl_divergence(p.ravel(), q.ravel())
    error = [kl_curr]

    for i in range(0, n_iters):
        kl_prev = kl_curr
        # Update c_x, q
        c_x = np.matrix(u.next_c_x(p, q, c_x, k))
        q = u.calc_q(p, range(0, m), c_x, range(0, n), c_y)

        # Update c_y, q
        c_y = np.matrix(u.next_c_y(p, q, c_y, l))
        q = u.calc_q(p, range(0, m), c_x, range(0, n), c_y)

        kl_curr = u.kl_divergence(p.ravel(), q.ravel())
        error.append(kl_curr)

        if verbose:
            print("iteration:    ", i, "Error:    ", kl_curr, "diff: ",
                  abs(kl_prev-kl_curr))

        if abs(kl_prev - kl_curr) < converge_threshold:
            break

    m1 = u.sorting(p, k, l, c_x, c_y)
    clustered = u.prob_clust(m1, range(0, k), c_x, range(0, l), c_y)

    return m1, q, c_x, c_y, clustered, error


def itcc_restricted(p, k, n_iters, converge_threshold, c_x, verbose=False):
    m = np.shape(p)[0]
    n = np.shape(p)[1]

    if m != n:
        raise Exception(f'Input must be square array, was shape {np.shape}.')

    c_y = np.copy(c_x)
    q = u.calc_q(p, range(0, m), c_x, range(0, n), c_y)
    kl_curr = u.kl_divergence(p.ravel(), q.ravel())
    error = [kl_curr]

    for i in range(0, n_iters):
        kl_prev = kl_curr
    # Update c_x
        c_x = np.matrix(u.next_c_x(p, q, c_x, k))

    # Update c_y, q
        c_y = np.copy(c_x)
        q = u.calc_q(p, range(0, m), c_x, range(0, n), c_y)

        kl_curr = u.kl_divergence(p.ravel(), q.ravel())
        error.append(kl_curr)

        if verbose:
            print("iteration:    ", i, "Error:    ", kl_curr, "diff: ",
                  abs(kl_prev-kl_curr))

        if abs(kl_prev - kl_curr) < converge_threshold:
            break

    m1 = u.sorting(p, k, k, c_x, c_y)
    clustered = u.prob_clust(m1, range(0, k), c_x, range(0, k), c_y)
    return m1, q, c_x, c_y, clustered, error


def information_cocluster(data, k, l, iterations=10):

    from_names = dict(enumerate(data.index.values))
    to_names = dict(enumerate(data.columns.values))

    n_iters = 30
    converge_thresh = 0.000001

    min_error = np.float('inf')
    ret_c_x = np.matrix(np.random.randint(k, size=len(from_names)))
    ret_c_y = np.matrix(np.random.randint(l, size=len(to_names)))

    for i in range(0, iterations):
        c_x = np.matrix(np.random.randint(k, size=len(from_names)))
        c_y = np.matrix(np.random.randint(l, size=len(to_names)))

        m1, q, c_x, c_y, clustered, error = itcc(data.values, k, l, n_iters,
                                                 converge_thresh, c_x, c_y)
        if error[-1] < min_error:
            print(f'Iteration {i}, new minimum {error[-1]}.')
            min_error = error[-1]
            ret_c_x = c_x
            ret_c_y = c_y
        else:
            print(f'Iteration {i}, worse')

    return (
        np.asarray(ret_c_x, dtype=int).squeeze(),
        np.asarray(ret_c_y, dtype=int).squeeze()
    )


def greedy_cocluster(data, x):
    # data: matrix
    # x: vector of cluster indices
    curr_index = 0
    curr_kl = calculate_kl_divergence(data, x, x)
    while curr_index < len(x):
        print(f'Testing index {curr_index}, in cluster {x[curr_index]}.')
        best_value, best_kl = find_best_value(data, x, curr_index)
        if best_kl < curr_kl:
            x[curr_index] = best_value
            curr_kl = best_kl
            print(f'Found better cluster: {best_value} ({curr_kl}).')
            curr_index = 0
        else:
            curr_index += 1

    return x, curr_kl


def find_best_value(data, x, index):
    best_kl = np.float('inf')
    old_value = x[index]
    best_value = old_value
    for cluster in range(len(set(x))):
        x[index] = cluster
        kl = calculate_kl_divergence(data, x, x)
        if kl < best_kl:
            best_kl = kl
            best_value = cluster

    x[index] = old_value
    return best_value, best_kl


def information_cocluster_restricted(data, k, iterations=10):
    from_names = dict(enumerate(data.index.values))

    n_iters = 30
    converge_thresh = 0.000001

    min_error = np.float('inf')
    ret_c_x = np.matrix(np.random.randint(k, size=len(from_names)))

    for i in range(0, iterations):
        c_x = np.matrix(np.random.randint(k, size=len(from_names)))

        m1, q, c_x, c_y, clustered, error = itcc_restricted(data.values, k,
                                                            n_iters,
                                                            converge_thresh,
                                                            c_x)
        kl_divergence = error[-1]
        print(f'Before greedy optimisation: {kl_divergence}.')
        c_x, kl_divergence = greedy_cocluster(
            data,
            np.asarray(c_x, dtype=int).squeeze()
        )
        if kl_divergence < min_error:
            print(f'Iteration {i}, new minimum {kl_divergence}.')
            min_error = kl_divergence
            ret_c_x = c_x
        else:
            print(f'Iteration {i}, worse')

    return ret_c_x


def calculate_kl_divergence(data, c_x, c_y):
    c_x = np.array([c_x])
    c_y = np.array([c_y])
    m = np.shape(data)[0]
    n = np.shape(data)[1]
    from src.util import calc_q, kl_divergence
    q = calc_q(data.values, range(0, m), c_x, range(0, n), c_y)
    return kl_divergence(data.values.ravel(), q.ravel())


def calculate_cluster_strength(data, clusters):
    old_kl = calculate_kl_divergence(data, clusters, clusters)
    values = []
    all_values = []
    for r in range(len(clusters)):
        old_value = clusters[r]
        total = 0.0
        all_values_part = []
        for cluster in set(clusters):
            clusters[r] = cluster
            new_kl = calculate_kl_divergence(data, clusters, clusters)
            total += new_kl - old_kl
            all_values_part.append(new_kl - old_kl)
        values.append(total / old_kl)
        all_values.append(all_values_part)
        clusters[r] = old_value
    return values, np.array(all_values)


def calculate_row_cluster_strength(data, row_clusters, column_clusters):
    old_kl = calculate_kl_divergence(data, row_clusters, column_clusters)
    values = []
    all_values = []
    for r in range(len(row_clusters)):
        old_value = row_clusters[r]
        total = 0.0
        all_values_part = []
        for cluster in set(row_clusters):
            row_clusters[r] = cluster
            new_kl = calculate_kl_divergence(data, row_clusters,
                                             column_clusters)
            total += new_kl - old_kl
            all_values_part.append(new_kl - old_kl)
        values.append(total / old_kl)
        all_values.append(all_values_part)
        row_clusters[r] = old_value
    return values, np.array(all_values)
