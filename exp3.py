import multiprocessing as mp
import simufunc
from tqdm import tqdm
import pandas as pd
from functools import partial
import numpy as np
import math

if __name__ == '__main__':
    pool = mp.Pool(processes=6)
    results = np.zeros([2, 6])
    N = 100
    m = math.ceil(N**(1/2))
    print("Processing M=", m)
    thes = [0.01, 0.1, 0.5, 1, 10, 50]
    with pool:
        for t in range(len(thes)):
            print("Processing theta=", thes[t])
            result = pool.map(partial(simufunc.experiment1, N=100, M=m, theta=thes[t]), 
                [i for i in range(100)])
            results[0, t] = np.mean([r[0] for r in result])
            results[1, t] = np.mean([r[1] for r in result])
            pd.DataFrame(results, columns=thes, index=["Cor_kernel", "Linear_reg"]).to_csv("result3.csv")