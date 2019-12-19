import multiprocessing
import scipy
import scipy.spatial
import argparse
import pickle
import os
import numpy as np

def get_metric(mat_A,  mat_B):
    def _distance_it(m):
        d = np.copy(m)

        for i in range(1, m.shape[0]):
            d[i,0] += d[i-1, 0]
            for j in range(1, m.shape[1]):
                d[i,j] += min([d[i-1, j], d[i, j-1], d[i-1, j-1]])

        return d[-1,-1]
    
    distances = scipy.spatial.distance.cdist(mat_A, mat_B, metric='cosine')
        
    return _distance_it(distances) / (distances.size)


N = 0
raw = {}

def computedtw(i):
    d = np.zeros(N)
    for j in range(i+1, N):
        d[j] = get_metric(get_mat(i), get_mat(j))
    print(i)
    return d
 
    
def get_mat(idx):
    return raw["total_contribution"][idx]   


def compute_dtw_mat(raw_, pool_size):
    global N
    global raw
    
    N = len(raw_["names"])
    dist_mat = np.zeros((N,N))
    raw = raw_
    
    pool = multiprocessing.Pool(pool_size)    
    start = 0
    end = N
    
    result = pool.map(computedtw, range(start,end))
    dist_mat[start:end] = result
    
    dist_mat = dist_mat + dist_mat.T
    return dist_mat
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computing DTW for a given raw heatmaps')
    parser.add_argument('-r', '--raw', default="None",
                        type=str, help="raw heatmaps data")
    parser.add_argument('-p', '--process', default=12,
                        type=int, help="number of processes to use while computing")
    
    args = parser.parse_args()
    
    raw_ = pickle.load(open(args.raw, "rb"))
    pool_size = int(args.process)
    
    print(f"Running on file: {args.raw} for {pool_size} threads")
    
    dtw = compute_dtw_mat(raw_, pool_size)
    
    # save results 
    path = os.path.split(args.raw)[0]
    output_file = os.path.join(path, "dtw_raw.pkl")
      
    pickle.dump({"dists": dtw}, open(output_file, "wb"))
    