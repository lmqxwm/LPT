import numpy as np
import pandas as pd
import math
from itertools import permutations, combinations
import statsmodels.api as sm
import multiprocessing as mp
from tqdm import tqdm
import scipy.stats as st
from numba import jit, njit
from sklearn.cluster import KMeans

def compute_G(Z, M = 10):
    '''Compute bin edges for continuous Z

    Args: 
        M: num of bins
    
    Returns:
        G: (np.array) [len(Z)] with element of {0,1,...,M-1}
    '''
    assert len(Z.shape) < 3
    if len(Z.shape) == 1:
        if np.min(Z) < np.max(Z):
            temp_Z = Z
        else:
            temp_Z = np.random.uniform(0, 1, Z.shape[0])
        # bins = np.linspace(np.min(temp_Z), np.max(temp_Z), M+1)
        # bins[0] -= 1 # pd.cut doesn't include left boundary value except add right=False
        # G = np.array(pd.cut(temp_Z, bins, labels=[x for x in range(M)]))
        percentiles = np.linspace(0, 100, M+1)
        bins = np.percentile(temp_Z, percentiles)
        bins[-1] += 1 # np.percentile doesn't include right boundary
        G = np.searchsorted(bins, temp_Z, side='right') - 1
    if len(Z.shape) == 2:
        kmeans = KMeans(n_clusters=M, random_state=0).fit(Z)
        G = kmeans.labels_
    return G

def discretize(X, M = 10):
    '''Discretize continuous variable X

    Args: 
        M: num of bins
    
    Returns:
        new_X: (np.array) discretized X with values of midpoints in all bins
    '''
    percentiles = np.linspace(0, 100, M+1)
    bins = np.percentile(X, percentiles)
    mid_bins = (bins[0:-1] + bins[1:]) / 2
    bins[-1] += 1 # np.percentile doesn't include right boundary
    G = np.searchsorted(bins, X, side='right') - 1
    new_X = np.array([mid_bins[x] for x in G])
    return new_X

def compute_T(X, Y, Z, G, M=10, cont_z=True, cont_xy=False):
    '''Compute testing statistic from 'Local permutation tests for conditional independence'

    Args:
        X(discrete), Y(discrete), Z(discrete or continuous): data
        G: categories of bins
        M: num of bins
        l1: number of possible values for X
        l2: number of possible values for Y
        cont: whether Z is continuous
    
    Returns:
        T: computed testing statistic
    '''

    if cont_xy:
        #MM = np.max([int(np.sqrt(M)), 1])
        MM = np.min([np.max([int(M), 1]), 50])
        XX = discretize(X, MM)
        YY = discretize(Y, MM)
        l1 = MM
        l2 = MM
        X_range = list(set(XX))
        Y_range = list(set(YY))
    else:
        XX = X.copy()
        YY = Y.copy()
        X_range = list(set(XX))
        Y_range = list(set(YY))
        l1 = len(set(X))
        l2 = len(set(Y))
    
    def phi(x, y, x_im, y_im, y_jm):
        '''Same \phi in paper'''
        return int(all([x_im == x, y_im == y])) - int(all([x_im == x, y_jm == y]))

    def kernel(ind):
        '''Same kernel in paper'''
        assert len(ind) == 4 
        perm = permutations(ind)
        h = 0
        for temp_ind in list(perm):
            for x in X_range:
                for y in Y_range:
                    h += phi(x, y, XX[temp_ind[0]], YY[temp_ind[0]], YY[temp_ind[1]]) * phi(x, y, XX[temp_ind[2]], YY[temp_ind[2]], YY[temp_ind[3]])
        h /= math.factorial(4)
        return h
    
    def weighted_kernel(ind, w1, w2):
        '''Same weighted kernel in paper'''
        assert len(ind) == 4 
        perm = permutations(ind)
        h = 0
        for temp_ind in list(perm):
            for x in X_range:
                for y in Y_range:
                    a = 1
                    if len(w1) > 0:
                        a *= 1 + len(np.where(np.array(XX[w1]) == x)[0])
                    if len(w2) > 0:
                        a *= 1 + len(np.where(np.array(YY[w2]) == y)[0])
                    h += phi(x, y, XX[temp_ind[0]], YY[temp_ind[0]], YY[temp_ind[1]]) \
                         * phi(x, y, XX[temp_ind[2]], YY[temp_ind[2]], YY[temp_ind[3]]) / a
        h /= math.factorial(4)
        return h
    
    if cont_z:
        T = 0
        for m in range(M):
            m_ind = list(np.where(G == m)[0])
            if len(m_ind) >= 4:
                n = len(m_ind)
                U = 0
                combs = combinations(m_ind, 4)
                combs = [x for x in combs]
                if len(combs) > 1000:
                    combs = [combs[i] for i in np.random.choice(len(combs), size=1000, replace=False)]
                count = 0
                for ind in combs:
                    if all([ind[0] < ind[1], ind[1] < ind[2], ind[2] < ind[3]]):
                        U += kernel(ind)
                        count += 1
                U /= count
                T += n * U
    else:
        T = 0
        for m in range(M):
            m_ind = list(np.where(G == m)[0])
            if len(m_ind) >= 4:
                tm = int((len(m_ind) - 4)/4)
                t1 = np.min([tm, l1])
                t2 = np.min([tm, l2])
                w = np.random.choice(m_ind, t1+t2)
                w1 = w[0:t1]
                w2 = w[t1:(t1+t2)]
                n = len(m_ind)
                m_ind = list(set(m_ind) - set(w1) - set(w2))
                assert len(m_ind) >= 4 
                U = 0
                combs = combinations(m_ind, 4)
                combs = [x for x in combs]
                if len(combs) > 1000:
                    combs = [combs[i] for i in np.random.choice(len(combs), size=1000, replace=False)]
                count = 0
                for ind in combs:
                    if all([ind[0] < ind[1], ind[1] < ind[2], ind[2] < ind[3]]):
                        U += weighted_kernel(ind, w1, w2)
                        
                # if count > 0:
                #     U /= count
                # if count == 0:
                #     print(m_ind)
                #     for ind in combs:
                #         print(ind)
                    
                U /= math.comb(2 * tm + 4, 2)
                wm = np.sqrt(np.min([n, l1]) * np.min([n, l2]))
                T += n * U * wm
    return T
        

def compute_T_linear(X, Y, Z):
    '''Compute testing statistic from linear regression Y~X+Z

    Args:
        X, Y, Z: data
    
    Returns:
        T: computed testing statistic
    '''
    XX = sm.add_constant(np.column_stack((X, Z)))
    #XX = np.column_stack((X, Z))
    mod = sm.OLS(Y, XX).fit()
    T = mod.params[1]
    return T


def compute_T_double(X, Y, Z):
    '''Compute testing statistic from double linear regression

    Args:
        X, Y, Z: data
    
    Returns:
        T: computed testing statistic
    '''
    ZZ = sm.add_constant(Z)
    #ZZ = Z
    modx = sm.OLS(X, ZZ).fit()
    mody = sm.OLS(Y, ZZ).fit()
    T = np.corrcoef(modx.resid, mody.resid)[0, 1]
    return T


def LPT(X, Y, Z, G, B=100, M=10, cont_z=True, \
        cont_xy=False, sub=0, perm="y"):
    '''Local permutation test

    Args:
        X(discrete), Y(discrete), Z(discrete or continuous): data
        G: categories of bins
        B: num of permutation tests
        M: num of bins
        l1: number of possible values for X
        l2: number of possible values for Y
        alpha: confidence level
        cont: whether Z is continuous
    
    Returns:
        p1: p-value for testing statistic from compute_T
        p2: p-value for testing statistic from compute_T_linear
    '''

    ZZ = discretize(Z, M)

    T_sam_linear_y = compute_T_linear(X, Y, ZZ)
    T_per_linear_y = np.zeros(B)
    T_sam_linear_y_z = compute_T_linear(X, Y, Z)
    T_per_linear_y_z = np.zeros(B)
    # T_per_linear_y_sub = np.zeros(B)

    T_sam_linear_x = compute_T_linear(Y, X, ZZ)
    T_per_linear_x = np.zeros(B)
    T_sam_linear_x_z = compute_T_linear(Y, X, Z)
    T_per_linear_x_z = np.zeros(B)
    # T_per_linear_x_sub = np.zeros(B)

    T_sam_double = compute_T_double(X, Y, ZZ)
    T_per_double = np.zeros(B)
    T_sam_double_z = compute_T_double(X, Y, Z)
    T_per_double_z = np.zeros(B)
    # T_per_double_sub = np.zeros(B)

    def perm_Y(YY, GG, ss, MM):
        '''Permutate YY within each bin GG==? for M bins, sub:sub-partition, with seed ss'''
        new_Y = YY.copy()
        for m in range(MM):
            m_ind = list(np.where(GG == m)[0])
            m_ind_new = m_ind.copy()
            #np.random.seed(ss); np.random.shuffle(m_ind_new)
            np.random.shuffle(m_ind_new)
            new_Y[m_ind] = new_Y[m_ind_new]
        return new_Y
    
    # def subpartition_G(GG, MM, sub, Z):
    #     subG = GG.copy()
    #     subG = subG * sub
    #     for m in range(MM):
    #         m_ind = list(np.where(GG == m)[0])
    #         if len(Z[m_ind]) >= sub:
    #             subG[m_ind] += compute_G(Z[m_ind], M=sub)
    #     return subG


    # subG = subpartition_G(G, M, sub, Z)

    for b in range(B):
        if perm == "y":
            new_Y = perm_Y(Y, G, b, M)
            T_per_linear_x[b] = compute_T_linear(new_Y, X, ZZ)
            T_per_linear_y[b] = compute_T_linear(X, new_Y, ZZ)
            T_per_double[b] = compute_T_double(X, new_Y, ZZ)

            T_per_linear_x_z[b] = compute_T_linear(new_Y, X, Z)
            T_per_linear_y_z[b] = compute_T_linear(X, new_Y, Z)
            T_per_double_z[b] = compute_T_double(X, new_Y, Z)
        elif perm == "x":
            new_X = perm_Y(X, G, b+B, M)
            T_per_linear_x[b] = compute_T_linear(Y, new_X, ZZ)
            T_per_linear_y[b] = compute_T_linear(new_X, Y, ZZ)
            T_per_double[b] = compute_T_double(new_X, Y, ZZ)

            T_per_linear_x_z[b] = compute_T_linear(Y, new_X, Z)
            T_per_linear_y_z[b] = compute_T_linear(new_X, Y, Z)
            T_per_double_z[b] = compute_T_double(new_X, Y, Z)
        else:
            raise ValueError("Non-existing permutating variable")
        
        
        # new_Y_sub = perm_Y(Y, subG, b, M*sub)
        # T_per_linear_x_sub[b] = compute_T_linear(new_Y_sub, X, ZZ)
        # T_per_linear_y_sub[b] = compute_T_linear(X, new_Y_sub, ZZ)
        # T_per_double_sub[b] = compute_T_double(X, new_Y_sub, ZZ)

    


    # N = len(G)
    # NN = np.random.poisson(lam=N/2, size=1)
    # if NN > N:
    #     p = 1.0
    #     #p_two = 1.0
    #     p_sub = 1.0
    # else:
    #     NN_ind  = np.random.choice(N, NN, replace=False)
    #     T_sam = compute_T(X[NN_ind], Y[NN_ind], Z[NN_ind], G[NN_ind], M, cont_z, cont_xy)
    #     T_per = np.zeros(B)
    #     # T_per_sub = np.zeros(B)
    #     for b in range(B):
    #         new_Y = perm_Y(Y[NN_ind], G[NN_ind], b, M)
    #         T_per[b] = compute_T(X[NN_ind], new_Y, Z[NN_ind], G[NN_ind], M, cont_z, cont_xy)
    #         # new_Y_sub = perm_Y(Y[NN_ind], subG[NN_ind], b, M*sub)
    #         # T_per_sub[b] = compute_T(X[NN_ind], new_Y_sub, Z[NN_ind], G[NN_ind], M, cont_z, cont_xy)

    #     p = (T_per >= T_sam).sum() / B
    #     # p_sub = (T_per_sub >= T_sam).sum() / B
    #     #p_two = (np.abs(T_per) >= np.abs(T_sam)).sum() / B # the same as p
    p_linear_x = (np.abs(T_per_linear_x) >= np.abs(T_sam_linear_x)).sum() / B
    p_linear_y = (np.abs(T_per_linear_y) >= np.abs(T_sam_linear_y)).sum() / B
    p_double = (np.abs(T_per_double) >= np.abs(T_sam_double)).sum() / B
    p_linear_x_z = (np.abs(T_per_linear_x_z) >= np.abs(T_sam_linear_x_z)).sum() / B
    p_linear_y_z = (np.abs(T_per_linear_y_z) >= np.abs(T_sam_linear_y_z)).sum() / B
    p_double_z = (np.abs(T_per_double_z) >= np.abs(T_sam_double_z)).sum() / B
    # p_linear_x_sub = (np.abs(T_per_linear_x_sub) >= np.abs(T_sam_linear_x)).sum() / B
    # p_linear_y_sub = (np.abs(T_per_linear_y_sub) >= np.abs(T_sam_linear_y)).sum() / B
    # p_double_sub = (np.abs(T_per_double_sub) >= np.abs(T_sam_double)).sum() / B

    # print("T_per_linear_x", T_per_linear_x[:10])
    # print("T_sam_linear_x", T_sam_linear_x)
    # print("T_per_linear_y", T_per_linear_y[:10])
    # print("T_sam_linear_y", T_sam_linear_y)
    # print("T_per_double", T_per_double[:10])
    # print("T_sam_double", T_sam_double)
    # print("T_per_linear_x_sub", T_per_linear_x_sub[:10])
    # print(T_per_linear_y_sub[:10])
    # print(T_per_double_sub[:10])


    #return int(p <= alpha), int(p_linear <= alpha)
    #return p, p_linear_x, p_linear_y, p_double, p_sub, p_linear_x_sub, p_linear_y_sub, p_double_sub
    return p_linear_x, p_linear_y, p_double, p_linear_x_z, p_linear_y_z, p_double_z





# both linear + dist 3*Z
# linear +dist +nonlinear x + 3x^2 +2 
# log x -2
# both nonlinear x + 3x^2 +2  log(x+1)-2

# 确定sub和m。from12

# 是否follow same steps





