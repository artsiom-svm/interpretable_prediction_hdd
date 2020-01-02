import multiprocessing
import scipy
import scipy.spatial
import argparse
import pickle
import os
import numpy as np
import glob


N = 0
m = 0
raw = []

def get_metric(mat_A,  mat_B):
    cut_a = np.abs(np.mean(mat_A)) + 3 * np.std(mat_A)
    cut_b = np.abs(np.mean(mat_B)) + 3 * np.std(mat_B)

    mat_A = np.sign(mat_A) * np.minimum(np.abs(mat_A), cut_a) / np.minimum(np.max(np.abs(mat_A)), cut_a)

    mat_A = mat_A.flatten()

    mat_B = np.sign(mat_B) * np.minimum(np.abs(mat_B), cut_b) / np.minimum(np.max(np.abs(mat_B)), cut_b)

    mat_B = mat_B.flatten()

    return scipy.spatial.distance.cosine(mat_A, mat_B)


def load_data(logdir):
    return [ pickle.load(open(f, "rb")) for f in
        glob.glob(f"{logdir}/*/heatmaps/raw.pkl")]


def get_mat(i, j):
    mat = raw[j]["total_contribution"][i]
    return mat

def compute_similarity(i):
    d = np.zeros((m, m))
    for j in range(0, m):
        for k in range(j + 1, m):
            d[j, k] = get_metric(get_mat(i, j), get_mat(i, k))
    return (d + d.T).reshape(-1)


def compute_similarity_mat(data, pool_size):
    global N,m
    global raw

    N = len(data[0]["names"])
    m = len(data)
    raw = data
    sim_mat = np.zeros((N, m * m))

    pool = multiprocessing.Pool(pool_size)
    start = 0
    end = N

    result = pool.map(compute_similarity, range(start, end))
    sim_mat[start:end] = result

    return sim_mat


def compute_stats(mat):
    return mat.std(axis=1), mat.mean(axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computing cosine similarity for a given logdir with heatmaps')
    parser.add_argument('-i', '--logdir', default="None",
                        type=str, help="top level with all runs to compute. Eg logs/fit/arch_name")
    parser.add_argument('-p', '--process', default=12,
                        type=int, help="number of processes to use while computing")

    args = parser.parse_args()

    data = load_data(args.logdir)
    pool_size = int(args.process)

    print(f"Computing similarity between {len(data)} runs with {pool_size} threads")

    similarity = compute_similarity_mat(data, pool_size)
    stds, means = compute_stats(similarity)

    # save results
    output_file = os.path.join(args.logdir, "cosine_similary.pkl")

    pickle.dump({
        "similarities": similarity,
        "stds": stds,
        "means": means
    }, open(output_file, "wb"))
