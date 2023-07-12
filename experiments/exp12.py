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
    results = np.zeros([8, 4])
    N = 100
    #Ms = [10]
    Ms = [2,5, 10, 25]

    types = ["normal", "normal_large", "normal_small", "skewed_normal"]
    # for t in range(len(types)):
    #     X, Y, Z = simufunc.data_generative6(type=types[t])
    #     print("Processing type=", types[t])
    #     print(X.shape, Y.shape, Z.shape)
    print("All M:", Ms)
    with pool:
        for m in range(len(Ms)):
            if Ms[m] <= 5:
                subs = [2, 4, 10]
            elif Ms[m] < 25:
                subs = [2, 4]
            else:
                subs=[2]
            for sub in subs:
                for t in range(len(types)):
                    print("Processing type=", types[t])
                    print("Processing M=", Ms[m])
                    print("Processing sub=", sub)
                    result = pool.map(partial(simufunc.experiment9, N=N, M=Ms[m], type=types[t], sub=sub), 
                        [i for i in range(100)])
                    results[0, t] = np.mean([r[0] for r in result])
                    results[1, t] = np.mean([r[1] for r in result])
                    results[2, t] = np.mean([r[2] for r in result])
                    results[3, t] = np.mean([r[3] for r in result])
                    results[4, t] = np.mean([r[4] for r in result])
                    results[5, t] = np.mean([r[5] for r in result])
                    results[6, t] = np.mean([r[6] for r in result])
                    results[7, t] = np.mean([r[7] for r in result])
                    pd.DataFrame(results, 
                        columns=types, 
                        index=["Cor_kernel", "Linear_reg_x", "Linear_reg_y", "Double_reg",
                        "double_Cor_kernel", "Linear_reg_x_sub", "Linear_reg_y_sub", "Double_reg_sub"]).to_csv(
                            sys.path[0]+"/results/result_sub/result_h1_M_"+str(Ms[m])+"_sub_"+str(sub)+".csv")

        print("======================processing h0")
        Ms = [2,5, 10, 25]
        results = np.zeros([8, 10])
        types = ["normal", "normal_large", "normal_small", "chi", "t", "exp", "uni", 
            "poi", "skewed_normal", "skewed_t"]
        for m in range(len(Ms)):
            if Ms[m] <= 5:
                subs = [2, 4, 10]
            elif Ms[m] < 25:
                subs = [2, 4]
            else:
                subs=[2]
            for sub in subs:
                for t in range(len(types)):
                    print("Processing type=", types[t])
                    print("Processing M=", Ms[m])
                    print("Processing sub=", sub)
                    result = pool.map(partial(simufunc.experiment99, N=N, M=Ms[m], type=types[t], sub=sub), 
                        [i for i in range(100)])
                    results[0, t] = np.mean([r[0] for r in result])
                    results[1, t] = np.mean([r[1] for r in result])
                    results[2, t] = np.mean([r[2] for r in result])
                    results[3, t] = np.mean([r[3] for r in result])
                    results[4, t] = np.mean([r[4] for r in result])
                    results[5, t] = np.mean([r[5] for r in result])
                    results[6, t] = np.mean([r[6] for r in result])
                    results[7, t] = np.mean([r[7] for r in result])
                    pd.DataFrame(results, 
                        columns=types, 
                        index=["Cor_kernel", "Linear_reg_x", "Linear_reg_y", "Double_reg",
                        "double_Cor_kernel", "Linear_reg_x_sub", "Linear_reg_y_sub", "Double_reg_sub"]).to_csv(
                            sys.path[0]+"/results/result_sub/result_h0_M_"+str(Ms[m])+"_sub_"+str(sub)+".csv")
    pool.close()