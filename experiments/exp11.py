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
    N = 100
    Nvars = [5, 10, 20]
    #Ms = [math.ceil(N**(1/10)), math.ceil(N**(1/4)), 8, math.ceil(N**(1/2)), 25, 50]
    m = 10
    with pool:
        for nvar in range(len(Nvars)):
        
            # result1 = np.zeros([nvar, nvar])
            # result2 = np.zeros([nvar, nvar])
            # result3 = np.zeros([nvar, nvar])
            # result4 = np.zeros([nvar, nvar])
            print("Processing Nvar=", Nvars[nvar])
            
            results = pool.map(partial(simufunc.experiment8, N=N, M=m, Nvar=Nvars[nvar]), 
                [i for i in range(150)])
            for i in range(4):
                result = np.mean([r[i] for r in results], axis=0)
                pd.DataFrame(result, 
                    columns=['X' + str(i) for i in range(1, Nvars[nvar]+1)], 
                    index=['X' + str(i) for i in range(1, Nvars[nvar]+1)]).to_csv(
                    sys.path[0]+"/results/result_markov_method_"+str(i)+"_Nvar_"+str(Nvars[nvar])+".csv")
            # result1 = np.mean([r[0] for r in result], axis=0)
                # result2 = np.mean([r[1] for r in result], axis=0)
                # result3 = np.mean([r[2] for r in result], axis=0)
                # result4 = np.mean([r[3] for r in result], axis=0)
    pool.close()