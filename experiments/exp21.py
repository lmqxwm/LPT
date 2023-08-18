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
    
    types = ["normal"]
    hs =  ["h0"] * 1
    assert len(types) == len(hs)
    print("All M:", Ms)
    with pool:
        print("===============================process one")
        xfuns = [None, expfunc.Z_to_Y]
        yfuns = [None, expfunc.Z_to_Y]
        vxs = [1, 0.1]
        vys = [1, 0.1]

        for t in range(len(types)):
            for xf in range(len(xfuns)):
                for yf in range(len(yfuns)):
                    for vx1 in range(len(vxs)):
                        for vy1 in range(len(vys)):
                            results = np.zeros([6, 6])
                            for m in range(len(Ms)):
                                print("Processing type=", types[t])
                                print("Processing M=", Ms[m])
                                result = pool.map(partial(expfunc.experiment16, 
                                            N=N, M=Ms[m], type=types[t], hypo=hs[t], cor=0.4,
                                            xfun=xfuns[xf], yfun=yfuns[yf], sub=2, perm="x",
                                            vx=vxs[vx1], vy=vys[vy1]), 
                                            [int(i+(xf+yf*2+vx1*5+vy1*7)*10) for i in range(100)])

               
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
                                        sys.path[0]+"/results/result_z/result_551_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_"+hs[t]+"_"+types[t]+".csv")

        print("===============================process one2")
        xfuns = [None, expfunc.Z_to_Y]
        yfuns = [None, expfunc.Z_to_Y]
        vxs = [5, 0.01]
        vys = [5, 0.01]

        for t in range(len(types)):
            for xf in range(len(xfuns)):
                for yf in range(len(yfuns)):
                    for vx1 in range(len(vxs)):
                        for vy1 in range(len(vys)):
                            results = np.zeros([6, 6])
                            for m in range(len(Ms)):
                                print("Processing type=", types[t])
                                print("Processing M=", Ms[m])
                                result = pool.map(partial(expfunc.experiment16, 
                                            N=N, M=Ms[m], type=types[t], hypo=hs[t], cor=0.4,
                                            xfun=xfuns[xf], yfun=yfuns[yf], sub=2, perm="x",
                                            vx=vxs[vx1], vy=vys[vy1]), 
                                            [int(i+(xf+yf*2+vx1*5+vy1*7)*20) for i in range(100)])

               
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
                                        sys.path[0]+"/results/result_z/result_552_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_"+hs[t]+"_"+types[t]+".csv")

        print("===============================process one3")
        xfuns = [None, expfunc.Z_to_Y]
        yfuns = [None, expfunc.Z_to_Y]
        vxs = [1, 0.01]
        vys = [1, 0.01]

        for t in range(len(types)):
            for xf in range(len(xfuns)):
                for yf in range(len(yfuns)):
                    for vx1 in range(len(vxs)):
                        for vy1 in range(len(vys)):
                            results = np.zeros([6, 6])
                            for m in range(len(Ms)):
                                print("Processing type=", types[t])
                                print("Processing M=", Ms[m])
                                result = pool.map(partial(expfunc.experiment16, 
                                            N=N, M=Ms[m], type=types[t], hypo=hs[t], cor=0.4,
                                            xfun=xfuns[xf], yfun=yfuns[yf], sub=2, perm="x",
                                            vx=vxs[vx1], vy=vys[vy1]), 
                                            [int(i+(xf+yf*2+vx1*5+vy1*7)*30) for i in range(100)])

               
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
                                        sys.path[0]+"/results/result_z/result_553_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_"+hs[t]+"_"+types[t]+".csv")


        # print("===============================process two")
        # xfuns = [None, expfunc.Z_to_Y4]
        # yfuns = [None, expfunc.Z_to_Y4]
        # vxs = [2, 0.1]
        # vys = [2, 0.1]

        # for t in range(len(types)):
        #     for xf in range(len(xfuns)):
        #         for yf in range(len(yfuns)):
        #             for vx1 in range(len(vxs)):
        #                 for vy1 in range(len(vys)):
        #                     results = np.zeros([6, 6])
        #                     for m in range(len(Ms)):
        #                         print("Processing type=", types[t])
        #                         print("Processing M=", Ms[m])
        #                         result = pool.map(partial(expfunc.experiment16, 
        #                                     N=N, M=Ms[m], type=types[t], hypo=hs[t], cor=0.4,
        #                                     xfun=xfuns[xf], yfun=yfuns[yf], sub=2, perm="x",
        #                                     vx=vxs[vx1], vy=vys[vy1]), 
        #                                     [int(i+(xf+yf*2+vx1*5+vy1*7)*2000) for i in range(100)])

               
        #                         results[0, m] = np.mean([r[0] for r in result])
        #                         results[1, m] = np.mean([r[1] for r in result])
        #                         results[2, m] = np.mean([r[2] for r in result])
        #                         results[3, m] = np.mean([r[3] for r in result])
        #                         results[4, m] = np.mean([r[4] for r in result])
        #                         results[5, m] = np.mean([r[5] for r in result])
        #                         pd.DataFrame(results, 
        #                             columns=Ms, 
        #                             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #                             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                                 sys.path[0]+"/results/result_z/result_66_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_"+hs[t]+"_"+types[t]+".csv")
        
        # print("===============================process three")
        # xfuns = [None, expfunc.Z_to_Y6]
        # yfuns = [None, expfunc.Z_to_Y6]
        # vxs = [1, 0.1]
        # vys = [1, 0.1]

        # for t in range(len(types)):
        #     for xf in range(len(xfuns)):
        #         for yf in range(len(yfuns)):
        #             for vx1 in range(len(vxs)):
        #                 for vy1 in range(len(vys)):
        #                     results = np.zeros([6, 6])
        #                     for m in range(len(Ms)):
        #                         print("Processing type=", types[t])
        #                         print("Processing M=", Ms[m])
        #                         result = pool.map(partial(expfunc.experiment16, 
        #                                     N=N, M=Ms[m], type=types[t], hypo=hs[t], cor=0.4,
        #                                     xfun=xfuns[xf], yfun=yfuns[yf], sub=2, perm="x",
        #                                     vx=vxs[vx1], vy=vys[vy1]), 
        #                                     [int(i+(xf+yf*2+vx1*5+vy1*7)*3000) for i in range(100)])

               
        #                         results[0, m] = np.mean([r[0] for r in result])
        #                         results[1, m] = np.mean([r[1] for r in result])
        #                         results[2, m] = np.mean([r[2] for r in result])
        #                         results[3, m] = np.mean([r[3] for r in result])
        #                         results[4, m] = np.mean([r[4] for r in result])
        #                         results[5, m] = np.mean([r[5] for r in result])
        #                         pd.DataFrame(results, 
        #                             columns=Ms, 
        #                             index=["Linear_reg_x", "Linear_reg_y", "Double_reg",
        #                             "Linear_reg_x_z", "Linear_reg_y_z", "Double_reg_z"]).to_csv(
        #                                 sys.path[0]+"/results/result_z/result_77_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_"+hs[t]+"_"+types[t]+".csv")




        pool.close()