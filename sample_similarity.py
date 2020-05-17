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

    mat_A = np.sign(mat_A) * np.minimum(np.abs(mat_A), cut_a) / \
        np.minimum(np.max(np.abs(mat_A)), cut_a)

    mat_A = mat_A.flatten()

    mat_B = np.sign(mat_B) * np.minimum(np.abs(mat_B), cut_b) / \
        np.minimum(np.max(np.abs(mat_B)), cut_b)

    mat_B = mat_B.flatten()

    return scipy.spatial.distance.cosine(mat_A, mat_B)


def load_data(fls):
    return [pickle.load(open(f"{f}/heatmaps/raw.pkl", "rb")) for f in
            fls]


def get_mat(i, j):
    mat = raw[j]["total_contribution"][i]
    return mat


def compute_similarity(i):
    while True:
        p = np.random.randint(0, len(raw))
        q = np.random.randint(0, len(raw))
        j = np.random.randint(0, N / len(raw) / len(raw))
        k = np.random.randint(0, N / len(raw) / len(raw))
        mat_A = get_mat(j, p)
        mat_B = get_mat(k, q)
        if j != k and mat_A.shape == mat_B.shape:
            return get_metric(mat_A, mat_B).reshape(-1, 1)


def compute_similarity_mat(data, pool_size):
    global N, m
    global raw

    N = len(data[0]["names"]) * len(data) * len(data)
    raw = data
    sim_mat = np.zeros((N, 1))

    pool = multiprocessing.Pool(pool_size)
    start = 0
    end = N

    result = pool.map(compute_similarity, range(start, end))
    sim_mat[start:end] = result

    return sim_mat


def compute_stats(mat):
    return mat.std(axis=1), mat.mean(axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Computing cosine similarity for a given logdir with heatmaps')
    parser.add_argument('-i', '--logdir', default="None",
                        type=str, help="top level with all runs to compute. Eg logs/fit/arch_name")
    parser.add_argument('-p', '--process', default=12,
                        type=int, help="number of processes to use while computing")

    args = parser.parse_args()
    fls = ["logs/fit/RETAIN_HDD/20191120-014535", "all other logs"]
    data = load_data(fls)
    pool_size = int(args.process)

    print(
        f"Computing similarity between {len(data)} runs with {pool_size} threads")

    similarity = compute_similarity_mat(data, pool_size)
    stds, means = compute_stats(similarity)

    # save results
    output_file = "sample_cosine_similary.pkl"

    pickle.dump({
        "similarities": similarity,
        "stds": stds,
        "means": means
    }, open(output_file, "wb"))
