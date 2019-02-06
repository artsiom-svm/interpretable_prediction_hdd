import pandas as pd
import numpy as np
import sys
import h5py


def data2mat(data):
    return None

def read_set(source, dataset, ids):
    for chunk in pd.read_csv(source, chunksize=chunksize):
        unique_drives = pd.unique(chunk.loc[:, 'drive_id'])
        if cur_drive and cur_drive in unique_drives:
            # special case to merge with data from previous chunk
            # get next frames
            data_cont = chunk.loc[chunk.drive_id == cur_drive, ids]
            # convert to mat
            mat_cont = data2mat(data_cont)
            # concatenate

            # remore from list
            unique_drives = unique_drives[unique_drives != drive]
        for drive in unique_drives:
            # slice data
            data = chunk.loc[chunk.drive_id == drive, ids]
            # convert to mat
            mat = data2mat(data)
            # push to dataset
            

        # remember the last driver id
        cur_drive = chunk.loc[-1, 'drive_id']
 

def process(source_dir, ids):
    chunksize = 10**6

    h5 = h5py.File("dataset_full.hdf5") 
    try:
        google = h5.create_group('google')
        error_log = google.create_dataset("errorlog", dtype=np.float64)
        badchips = google.create_dataset("badchips", dtype=np.float64)
        swaplog = google.create_dataset("swaplog", dtype=np.float64)
    except:
        pass

    read_set(f"{source_dir}/errorlog_fixed.csv", h5['google/errorlog'], ids)   
    read_set(f"{source_dir}/badchips.csv", h5['google/badchips'], ids)   
    read_set(f"{source_dir}/swaplog.csv", h5['google/swaplog'], ids)   


if __name__ == "__main__":
    source = sys.argv[1]
    ids = open(sys.argv[2], 'r').read().split()
    process(source, ids)