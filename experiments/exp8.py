import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
from functools import partial
import numpy as np
import math
import sys 
sys.path.insert(0, sys.path[0]+"/../")
import simufunc
import os

if __name__ == '__main__':
    pool = mp.Pool(processes=6)
    results = np.zeros([3, 1])
    N = 100
    Ms = [10]
    #Ms = [math.ceil(N**(1/10)), math.ceil(N**(1/4)), 6, math.ceil(N**(1/2)), 25, 50]
    print("All M:", Ms)
    with pool:
        for m in range(len(Ms)):
            print("Processing M=", Ms[m])
            result = pool.map(partial(simufunc.experiment6, N=N, M=Ms[m]), 
                [i for i in range(100)])
            results[0, m] = np.mean([r[0] for r in result])
            results[1, m] = np.mean([r[1] for r in result])
            results[2, m] = np.mean([r[2] for r in result])
            pd.DataFrame(results, 
                columns=Ms, 
                index=["Cor_kernel", "Linear_reg", "Double_reg"]).to_csv(
                    sys.path[0]+"/results/result8.csv")