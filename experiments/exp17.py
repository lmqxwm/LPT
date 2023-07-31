import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
from functools import partial
import numpy as np
import math
import sys 
sys.path.insert(0, sys.path[0]+"/../")
import os
import expfunc

if __name__ == '__main__':
    pool = mp.Pool(processes=6)
    #results = np.zeros([4, 5])
    N = 100
    #Ms = [10]
    Ms = [math.ceil(N**(1/10)), 5, math.ceil(N**(1/2)), 20]

    types = ["normal", "skewed_normal",
        "normal", "uni", 
        "poi", "skewed_normal"]
    hs =  ["h1"] * 2 + ["h0"] * 4
    assert len(types) == len(hs)
    # types = ["normal", "normal_large", "normal_small", "chi", "t", "exp", "uni", 
    #     "poi", "skewed_normal", "skewed_t"]
    # for t in range(len(types)):
    #     X, Y, Z = simufunc.data_generative8(type=types[t], hypo="h0", yfun=simufunc.Z_to_Y)
    #     print("Processing type=", types[t])
    #     print(X.shape, Y.shape, Z.shape)

    print("All M:", Ms)
    with pool:
        results = np.zeros([8, 4])
        for t in range(len(types)):
            for m in range(len(Ms)):
                print("Processing type=", types[t])
                print("Processing M=", Ms[m])
                result = pool.map(partial(expfunc.experiment13, 
                    N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=None, yfun=expfunc.Z_to_Y8, sub=4), 
                    [i+100 for i in range(100)])
                results[0, m] = np.mean([r[0] for r in result])
                results[1, m] = np.mean([r[1] for r in result])
                results[2, m] = np.mean([r[2] for r in result])
                results[3, m] = np.mean([r[3] for r in result])
                results[4, m] = np.mean([r[4] for r in result])
                results[5, m] = np.mean([r[5] for r in result])
                results[6, m] = np.mean([r[6] for r in result])
                results[7, m] = np.mean([r[7] for r in result])
                pd.DataFrame(results, 
                    columns=Ms, 
                    index=["Cor_kernel", "Linear_reg_x", "Linear_reg_y", "Double_reg",
                    "double_Cor_kernel", "Linear_reg_x_sub", "Linear_reg_y_sub", "Double_reg_sub"]).to_csv(
                        sys.path[0]+"/results/result_smooth/result_func38_"+hs[t]+"_"+types[t]+".csv")
        
    pool.close()