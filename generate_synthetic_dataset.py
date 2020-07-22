import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import blimpy as bl
import pandas as pd
from astropy import units as u

import sys, os, glob, errno
import csv
import json
import h5py
import time

sys.path.insert(0, "/home/bryanb/setigen/")
import setigen as stg



def signal_db_heuristic():
    # Defining a heuristic that follows the empirical distribution found in Price et al. 2019
    x = np.linspace(10, 50, num=100, endpoint=False)
    y = 1 - (x/50)**2
    y1 = np.power(10, y)
    
    xk = x
    pk = y1 / np.sum(y1)
    return xk, pk


def generate_frame(snr):
    frame = stg.Frame(fchans=512,
                      tchans=16,
                      df=2.7939677238464355*u.Hz,
                      dt=18.25361108*u.s,
                      fch1=6095.214842353016*u.MHz)
    
    frame.add_noise_from_obs(share_index=True)
    noise_mean, noise_std = frame.get_noise_stats()

    start_index = np.random.randint(0, frame.fchans)
    end_index = np.random.randint(0, frame.fchans)
    drift_rate = frame.get_drift_rate(start_index, end_index)

    width = np.random.uniform(10, 30)

    signal = frame.add_constant_signal(f_start=frame.get_frequency(start_index),
                              drift_rate=drift_rate*u.Hz/u.s,
                              level=frame.get_intensity(snr=snr),
                              width=width*u.Hz,
                              f_profile_type='gaussian')

    frame_info = {
        'noise_mean': noise_mean,
        'noise_std': noise_std,
        'snr': snr,
        'start_index': start_index,
        'end_index': end_index,
        'width': width
    }
    return frame, frame_info


def db_to_snr(db):
    return np.power(10, db / 10)


def mkdir(d):
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def generate_from_pdf(xk, pk):
    path = '/datax/scratch/bbrzycki/data/synthetic_dataset/emp'
    mkdir(path)
    
    for folder, size in [('train', 60000), ('test', 6000)]:
        mkdir('{}/{}'.format(path, folder))
        
        info_list = []
        for i in range(size):
            fn = 'emp_{:06d}.npy'.format(i)
    
            snr = db_to_snr(np.random.choice(xk, p=pk))
            frame, frame_info = generate_frame(snr)
            frame.save_npy('{}/{}/{}'.format(path, folder, fn))
            
            frame_info['filename'] = fn 
            info_list.append(frame_info)
            
            print('Saved frame {} in {}'.format(i, folder))
            
        df = pd.DataFrame(info_list)
        df.to_csv('{}/{}/labels.csv'.format(path, folder), index=False)
            
            


def generate_from_uniform():
    path = '/datax/scratch/bbrzycki/data/synthetic_dataset/low'
    mkdir(path)
    
    for folder, split_size in [('train', 10000), ('test', 1000)]:
        mkdir('{}/{}'.format(path, folder))
        
        info_list = []
        for db in np.linspace(0, 25, 6):
            snr = db_to_snr(db)
            
            for i in range(split_size):
                fn = 'low_{:02d}db_{:06d}.npy'.format(int(db), i)
    
                frame, frame_info = generate_frame(snr)
                frame.save_npy('{}/{}/{}'.format(path, folder, fn))

                frame_info['filename'] = fn 
                info_list.append(frame_info)
                
                print('Saved frame {} for db {} in {}'.format(i, int(db), folder))
            
        df = pd.DataFrame(info_list)
        df.to_csv('{}/{}/labels.csv'.format(path, folder), index=False)
        




if __name__ == '__main__':
    start = time.time()
    mkdir('/datax/scratch/bbrzycki/data/synthetic_dataset')
    
    xk, pk = signal_db_heuristic()
    generate_from_pdf(xk, pk)
    
    emp_time = time.time()
    
    generate_from_uniform()
    
    low_time = time.time()
    
    print('emp_time:', emp_time - start)
    print('low_time:', low_time - emp_time)