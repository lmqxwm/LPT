import multiprocessing as mp
import simufunc
from tqdm import tqdm
import pandas as pd
from functools import partial
import numpy as np
import math

if __name__ == '__main__':
    pool = mp.Pool(processes=6)
    results = np.zeros([2, 4])
    N = 100
    Ms = [math.ceil(N**(1/10)), math.ceil(N**(1/4)), math.ceil(N**(1/2)), N]
    print("All M:", Ms)
    with pool:
        for m in range(len(Ms)):
            print("Processing M=", Ms[m])
            result = pool.map(partial(simufunc.experiment2, N=100, M=Ms[m]), 
                [i for i in range(100)])
            results[0, m] = np.mean([r[0] for r in result])
            results[1, m] = np.mean([r[1] for r in result])
            pd.DataFrame(results, columns=Ms, index=["Cor_kernel", "Linear_reg"]).to_csv("result2.csv")
    #pd.DataFrame({"Cor_kernel": [r[0] for r in result],
    #    "Linear_reg":[r[1] for r in result]}).to_csv("result3.csv", index=False)

    
    #[pool.apply_async(simufunc.experiment3, (i,)) for i in range(100)]