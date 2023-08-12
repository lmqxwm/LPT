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
import simufunc

if __name__ == '__main__':
    pool = mp.Pool(processes=6)
    #results = np.zeros([4, 5])
    N = 100
    #Ms = [10]
    Ms = [math.ceil(N**(1/10)), 5, math.ceil(N**(1/2)), 16, 25, 50]

    #types = ["normal", "skewed_normal", "normal", "skewed_normal", "uni", "poi"]
    
    types = ["normal", "skewed_normal", "normal", "skewed_normal"]
    hs =  ["h1"] * 2 + ["h0"] * 2
    assert len(types) == len(hs)
    print("All M:", Ms)
    with pool:

        # print("===========================process one")
        # results = np.zeros([6, 6])
        # for t in range(len(types)):
        #     for m in range(len(Ms)):
        #         print("Processing type=", types[t])
        #         print("Processing M=", Ms[m])
        #         result = pool.map(partial(expfunc.experiment15, 
        #             N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=None, yfun=None, sub=2, perm="y"), 
        #             [i+100 for i in range(100)])
        #         results[0, m] = np.mean([r[0] for r in result])
        #         results[1, m] = np.mean([r[1] for r in result])
        #         results[2, m] = np.mean([r[2] for r in result])
        #         results[3, m] = np.mean([r[3] for r in result])
        #         results[4, m] = np.mean([r[4] for r in result])
        #         results[5, m] = np.mean([r[5] for r in result])
        #         pd.DataFrame(results, 
        #             columns=Ms, 
        #             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                 sys.path[0]+"/results/result_z/result_2_func00_ss_y_"+hs[t]+"_"+types[t]+".csv")
        
        # print("===========================process one 2")
        # results = np.zeros([6, 6])
        # for t in range(len(types)):
        #     for m in range(len(Ms)):
        #         print("Processing type=", types[t])
        #         print("Processing M=", Ms[m])
        #         result = pool.map(partial(expfunc.experiment15, 
        #             N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=None, yfun=None, sub=2, perm="x"), 
        #             [i+200 for i in range(100)])
        #         results[0, m] = np.mean([r[0] for r in result])
        #         results[1, m] = np.mean([r[1] for r in result])
        #         results[2, m] = np.mean([r[2] for r in result])
        #         results[3, m] = np.mean([r[3] for r in result])
        #         results[4, m] = np.mean([r[4] for r in result])
        #         results[5, m] = np.mean([r[5] for r in result])
        #         pd.DataFrame(results, 
        #             columns=Ms, 
        #             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                 sys.path[0]+"/results/result_z/result_2_func00_ss_x_"+hs[t]+"_"+types[t]+".csv")
        
        types = ["normal", "skewed_normal"]
        hs =  ["h1"] * 2 

        cors = [0.01, 0.05, 0.1, 0.15, 0.2]
        print("===========================process two")
        for c in range(len(cors)):
            print("============================process cor=", cors[c])
            results = np.zeros([6, 6])
            for t in range(len(types)):
                for m in range(len(Ms)):
                    print("Processing type=", types[t])
                    print("Processing M=", Ms[m])
                    result = pool.map(partial(expfunc.experiment15, 
                        N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=None, yfun=None, sub=2, perm="y", cor=cors[c]), 
                        [i+3000*c for i in range(100)])
                    results[0, m] = np.mean([r[0] for r in result])
                    results[1, m] = np.mean([r[1] for r in result])
                    results[2, m] = np.mean([r[2] for r in result])
                    results[3, m] = np.mean([r[3] for r in result])
                    results[4, m] = np.mean([r[4] for r in result])
                    results[5, m] = np.mean([r[5] for r in result])
                    pd.DataFrame(results, 
                        columns=Ms, 
                        index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
                        "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
                            sys.path[0]+"/results/result_z/result_3_func00_ss_y_"+hs[t]+"_"+types[t]+"_"+str(c)+".csv")
            
        
        print("===========================process three")
        for c in range(len(cors)):
            print("============================process cor=", cors[c])
            results = np.zeros([6, 6])
            for t in range(len(types)):
                for m in range(len(Ms)):
                    print("Processing type=", types[t])
                    print("Processing M=", Ms[m])
                    result = pool.map(partial(expfunc.experiment15, 
                        N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=expfunc.Z_to_Y4, yfun=expfunc.Z_to_Y7, sub=2, perm="y", cor=cors[c]), 
                        [i+4000*c for i in range(100)])
                    results[0, m] = np.mean([r[0] for r in result])
                    results[1, m] = np.mean([r[1] for r in result])
                    results[2, m] = np.mean([r[2] for r in result])
                    results[3, m] = np.mean([r[3] for r in result])
                    results[4, m] = np.mean([r[4] for r in result])
                    results[5, m] = np.mean([r[5] for r in result])
                    pd.DataFrame(results, 
                        columns=Ms, 
                        index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
                        "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
                            sys.path[0]+"/results/result_z/result_3_func47_ss_y_"+hs[t]+"_"+types[t]+"_"+str(c)+".csv")
        
        # types = ["normal", "skewed_normal", "normal", "skewed_normal"]
        # hs =  ["h1"] * 2 + ["h0"] * 2

        # print("===========================process four y")
        # results = np.zeros([6, 6])
        # for t in range(len(types)):
        #     for m in range(len(Ms)):
        #         print("Processing type=", types[t])
        #         print("Processing M=", Ms[m])
        #         result = pool.map(partial(expfunc.experiment14, 
        #             N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=None, yfun=None, sub=2, perm="y"), 
        #             [i+500 for i in range(100)])
        #         results[0, m] = np.mean([r[0] for r in result])
        #         results[1, m] = np.mean([r[1] for r in result])
        #         results[2, m] = np.mean([r[2] for r in result])
        #         results[3, m] = np.mean([r[3] for r in result])
        #         results[4, m] = np.mean([r[4] for r in result])
        #         results[5, m] = np.mean([r[5] for r in result])
        #         pd.DataFrame(results, 
        #             columns=Ms, 
        #             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                 sys.path[0]+"/results/result_z/result_2_func00_nn_y_"+hs[t]+"_"+types[t]+".csv")
        
        # print("===========================process four x")
        # results = np.zeros([6, 6])
        # for t in range(len(types)):
        #     for m in range(len(Ms)):
        #         print("Processing type=", types[t])
        #         print("Processing M=", Ms[m])
        #         result = pool.map(partial(expfunc.experiment14, 
        #             N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=None, yfun=None, sub=2, perm="x"), 
        #             [i+600 for i in range(100)])
        #         results[0, m] = np.mean([r[0] for r in result])
        #         results[1, m] = np.mean([r[1] for r in result])
        #         results[2, m] = np.mean([r[2] for r in result])
        #         results[3, m] = np.mean([r[3] for r in result])
        #         results[4, m] = np.mean([r[4] for r in result])
        #         results[5, m] = np.mean([r[5] for r in result])
        #         pd.DataFrame(results, 
        #             columns=Ms, 
        #             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                 sys.path[0]+"/results/result_z/result_2_func00_nn_x_"+hs[t]+"_"+types[t]+".csv")

        # print("===========================process five y")
        # results = np.zeros([6, 6])
        # for t in range(len(types)):
        #     for m in range(len(Ms)):
        #         print("Processing type=", types[t])
        #         print("Processing M=", Ms[m])
        #         result = pool.map(partial(expfunc.experiment14, 
        #             N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=expfunc.Z_to_Y4, yfun=expfunc.Z_to_Y7, sub=2, perm="y"), 
        #             [i+700 for i in range(100)])
        #         results[0, m] = np.mean([r[0] for r in result])
        #         results[1, m] = np.mean([r[1] for r in result])
        #         results[2, m] = np.mean([r[2] for r in result])
        #         results[3, m] = np.mean([r[3] for r in result])
        #         results[4, m] = np.mean([r[4] for r in result])
        #         results[5, m] = np.mean([r[5] for r in result])
        #         pd.DataFrame(results, 
        #             columns=Ms, 
        #             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                 sys.path[0]+"/results/result_z/result_2_func47_nn_y_"+hs[t]+"_"+types[t]+".csv")
        
        # print("===========================process five x")
        # results = np.zeros([6, 6])
        # for t in range(len(types)):
        #     for m in range(len(Ms)):
        #         print("Processing type=", types[t])
        #         print("Processing M=", Ms[m])
        #         result = pool.map(partial(expfunc.experiment14, 
        #             N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=expfunc.Z_to_Y4, yfun=expfunc.Z_to_Y7, sub=2, perm="x"), 
        #             [i+800 for i in range(100)])
        #         results[0, m] = np.mean([r[0] for r in result])
        #         results[1, m] = np.mean([r[1] for r in result])
        #         results[2, m] = np.mean([r[2] for r in result])
        #         results[3, m] = np.mean([r[3] for r in result])
        #         results[4, m] = np.mean([r[4] for r in result])
        #         results[5, m] = np.mean([r[5] for r in result])
        #         pd.DataFrame(results, 
        #             columns=Ms, 
        #             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                 sys.path[0]+"/results/result_z/result_2_func47_nn_x_"+hs[t]+"_"+types[t]+".csv")
        
        # print("===========================process six y")
        # results = np.zeros([6, 6])
        # for t in range(len(types)):
        #     for m in range(len(Ms)):
        #         print("Processing type=", types[t])
        #         print("Processing M=", Ms[m])
        #         result = pool.map(partial(expfunc.experiment13, 
        #             N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=None, yfun=None, sub=2, perm="y"), 
        #             [i+900 for i in range(100)])
        #         results[0, m] = np.mean([r[0] for r in result])
        #         results[1, m] = np.mean([r[1] for r in result])
        #         results[2, m] = np.mean([r[2] for r in result])
        #         results[3, m] = np.mean([r[3] for r in result])
        #         results[4, m] = np.mean([r[4] for r in result])
        #         results[5, m] = np.mean([r[5] for r in result])
        #         pd.DataFrame(results, 
        #             columns=Ms, 
        #             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                 sys.path[0]+"/results/result_z/result_2_func00_sn_y_"+hs[t]+"_"+types[t]+".csv")
        
        # print("===========================process six x")
        # results = np.zeros([6, 6])
        # for t in range(len(types)):
        #     for m in range(len(Ms)):
        #         print("Processing type=", types[t])
        #         print("Processing M=", Ms[m])
        #         result = pool.map(partial(expfunc.experiment13, 
        #             N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=None, yfun=None, sub=2, perm="x"), 
        #             [i+1000 for i in range(100)])
        #         results[0, m] = np.mean([r[0] for r in result])
        #         results[1, m] = np.mean([r[1] for r in result])
        #         results[2, m] = np.mean([r[2] for r in result])
        #         results[3, m] = np.mean([r[3] for r in result])
        #         results[4, m] = np.mean([r[4] for r in result])
        #         results[5, m] = np.mean([r[5] for r in result])
        #         pd.DataFrame(results, 
        #             columns=Ms, 
        #             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                 sys.path[0]+"/results/result_z/result_2_func00_sn_x_"+hs[t]+"_"+types[t]+".csv")
        
        # print("===========================process seven y")
        # results = np.zeros([6, 6])
        # for t in range(len(types)):
        #     for m in range(len(Ms)):
        #         print("Processing type=", types[t])
        #         print("Processing M=", Ms[m])
        #         result = pool.map(partial(expfunc.experiment13, 
        #             N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=expfunc.Z_to_Y4, yfun=expfunc.Z_to_Y7, sub=2, perm="y"), 
        #             [i+1100 for i in range(100)])
        #         results[0, m] = np.mean([r[0] for r in result])
        #         results[1, m] = np.mean([r[1] for r in result])
        #         results[2, m] = np.mean([r[2] for r in result])
        #         results[3, m] = np.mean([r[3] for r in result])
        #         results[4, m] = np.mean([r[4] for r in result])
        #         results[5, m] = np.mean([r[5] for r in result])
        #         pd.DataFrame(results, 
        #             columns=Ms, 
        #             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                 sys.path[0]+"/results/result_z/result_2_func47_sn_y_"+hs[t]+"_"+types[t]+".csv")

        # print("===========================process seven x")
        # results = np.zeros([6, 6])
        # for t in range(len(types)):
        #     for m in range(len(Ms)):
        #         print("Processing type=", types[t])
        #         print("Processing M=", Ms[m])
        #         result = pool.map(partial(expfunc.experiment13, 
        #             N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=expfunc.Z_to_Y4, yfun=expfunc.Z_to_Y7, sub=2, perm="x"), 
        #             [i+1200 for i in range(100)])
        #         results[0, m] = np.mean([r[0] for r in result])
        #         results[1, m] = np.mean([r[1] for r in result])
        #         results[2, m] = np.mean([r[2] for r in result])
        #         results[3, m] = np.mean([r[3] for r in result])
        #         results[4, m] = np.mean([r[4] for r in result])
        #         results[5, m] = np.mean([r[5] for r in result])
        #         pd.DataFrame(results, 
        #             columns=Ms, 
        #             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                 sys.path[0]+"/results/result_z/result_2_func47_sn_x_"+hs[t]+"_"+types[t]+".csv")
        
        # print("===========================process eight y")
        # results = np.zeros([6, 6])
        # for t in range(len(types)):
        #     for m in range(len(Ms)):
        #         print("Processing type=", types[t])
        #         print("Processing M=", Ms[m])
        #         result = pool.map(partial(expfunc.experiment13, 
        #             N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=expfunc.Z_to_Y3, yfun=expfunc.Z_to_Y3, sub=2, perm="y"), 
        #             [i+1300 for i in range(100)])
        #         results[0, m] = np.mean([r[0] for r in result])
        #         results[1, m] = np.mean([r[1] for r in result])
        #         results[2, m] = np.mean([r[2] for r in result])
        #         results[3, m] = np.mean([r[3] for r in result])
        #         results[4, m] = np.mean([r[4] for r in result])
        #         results[5, m] = np.mean([r[5] for r in result])
        #         pd.DataFrame(results, 
        #             columns=Ms, 
        #             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                 sys.path[0]+"/results/result_z/result_2_func33_sn_y_"+hs[t]+"_"+types[t]+".csv")

        # print("===========================process eight x")
        # results = np.zeros([6, 6])
        # for t in range(len(types)):
        #     for m in range(len(Ms)):
        #         print("Processing type=", types[t])
        #         print("Processing M=", Ms[m])
        #         result = pool.map(partial(expfunc.experiment13, 
        #             N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=expfunc.Z_to_Y3, yfun=expfunc.Z_to_Y3, sub=2, perm="x"), 
        #             [i+1400 for i in range(100)])
        #         results[0, m] = np.mean([r[0] for r in result])
        #         results[1, m] = np.mean([r[1] for r in result])
        #         results[2, m] = np.mean([r[2] for r in result])
        #         results[3, m] = np.mean([r[3] for r in result])
        #         results[4, m] = np.mean([r[4] for r in result])
        #         results[5, m] = np.mean([r[5] for r in result])
        #         pd.DataFrame(results, 
        #             columns=Ms, 
        #             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                 sys.path[0]+"/results/result_z/result_2_func33_sn_x_"+hs[t]+"_"+types[t]+".csv")
        
        #         print("===========================process nine y")
        # results = np.zeros([6, 6])
        # for t in range(len(types)):
        #     for m in range(len(Ms)):
        #         print("Processing type=", types[t])
        #         print("Processing M=", Ms[m])
        #         result = pool.map(partial(expfunc.experiment13, 
        #             N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=expfunc.Z_to_Y6, yfun=expfunc.Z_to_Y, sub=2, perm="y"), 
        #             [i+1500 for i in range(100)])
        #         results[0, m] = np.mean([r[0] for r in result])
        #         results[1, m] = np.mean([r[1] for r in result])
        #         results[2, m] = np.mean([r[2] for r in result])
        #         results[3, m] = np.mean([r[3] for r in result])
        #         results[4, m] = np.mean([r[4] for r in result])
        #         results[5, m] = np.mean([r[5] for r in result])
        #         pd.DataFrame(results, 
        #             columns=Ms, 
        #             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                 sys.path[0]+"/results/result_z/result_2_func61_sn_y_"+hs[t]+"_"+types[t]+".csv")

        # print("===========================process nine x")
        # results = np.zeros([6, 6])
        # for t in range(len(types)):
        #     for m in range(len(Ms)):
        #         print("Processing type=", types[t])
        #         print("Processing M=", Ms[m])
        #         result = pool.map(partial(expfunc.experiment13, 
        #             N=N, M=Ms[m], type=types[t], hypo=hs[t], xfun=expfunc.Z_to_Y6, yfun=expfunc.Z_to_Y, sub=2, perm="x"), 
        #             [i+1600 for i in range(100)])
        #         results[0, m] = np.mean([r[0] for r in result])
        #         results[1, m] = np.mean([r[1] for r in result])
        #         results[2, m] = np.mean([r[2] for r in result])
        #         results[3, m] = np.mean([r[3] for r in result])
        #         results[4, m] = np.mean([r[4] for r in result])
        #         results[5, m] = np.mean([r[5] for r in result])
        #         pd.DataFrame(results, 
        #             columns=Ms, 
        #             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                 sys.path[0]+"/results/result_z/result_2_func61_sn_x_"+hs[t]+"_"+types[t]+".csv")
        
        
    pool.close()
