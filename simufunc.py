import numpy as np
import pandas as pd
import math
from itertools import permutations, combinations
import statsmodels.formula.api as smf
import multiprocessing as mp
from tqdm import tqdm

def compute_G(Z, M = 10):
    '''Compute bin edges for continuous Z

    Args: 
        M: num of bins
    
    Returns:
        G: (np.array) [len(Z)] with element of {0,1,...,M-1}
    '''
    bins = np.linspace(np.min(Z), np.max(Z), M+1)
    bins[0] -= 1 # pd.cut doesn't include left boundary value except add right=False
    G = np.array(pd.cut(Z, bins, labels=[x for x in range(M)]))
    return G

def compute_T(X, Y, Z, G, M=10, l1=2, l2=2, cont=True):
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
    
    def phi(x, y, x_im, y_im, y_jm):
        '''Same \phi in paper'''
        return int(all([x_im == x, y_im == y])) - int(all([x_im == x, y_jm == y]))

    def kernel(ind):
        '''Same kernel in paper'''
        assert len(ind) == 4 
        perm = permutations(ind)
        h = 0
        for temp_ind in list(perm):
            for x in range(l1):
                for y in range(l2):
                    h += phi(x, y, X[temp_ind[0]], Y[temp_ind[0]], Y[temp_ind[1]]) * phi(x, y, X[temp_ind[2]], Y[temp_ind[2]], Y[temp_ind[3]])
        h /= math.factorial(4)
        return h
    
    def weighted_kernel(ind, w1, w2):
        '''Same weighted kernel in paper'''
        assert len(ind) == 4 
        perm = permutations(ind)
        h = 0
        for temp_ind in list(perm):
            for x in range(l1):
                for y in range(l2):
                    a = 1
                    if len(w1) > 0:
                        a *= 1 + len(np.where(np.array(w1) == x)[0])
                    if len(w2) > 0:
                        a *= 1 + len(np.where(np.array(w2) == y)[0])
                    h += phi(x, y, X[temp_ind[0]], Y[temp_ind[0]], Y[temp_ind[1]]) \
                         * phi(x, y, X[temp_ind[2]], Y[temp_ind[2]], Y[temp_ind[3]]) / a
        h /= math.factorial(4)
        return h
    
    if cont:
        T = 0
        for m in range(M):
            m_ind = list(np.where(G == m)[0])
            if len(m_ind) >= 4:
                n = len(m_ind)
                U = 0
                combs = combinations(m_ind, 4)
                combs = [x for x in combs]
                if len(combs) > 10000:
                    combs = [combs[i] for i in np.random.choice(len(combs), size=10000, replace=False)]
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
                # if len(combs) > 10000:
                #     combs = [combs[i] for i in np.random.choice(len(combs), size=10000, replace=False)]
                # count = 0
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
    '''Compute testing statistic from linear regression

    Args:
        X(discrete), Y(discrete), Z(discrete or continuous): data
    
    Returns:
        T: computed testing statistic
    '''
    mod = smf.ols(formula='Y ~ X + Z', data=pd.DataFrame({'X':X, 'Y':Y, 'Z':Z})).fit()
    T = mod.params['X']
    return T

def LPT(X, Y, Z, G, B=100, M=10, l1=2, l2=2, alpha=0.05, cont=True):
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
    T_sam = compute_T(X, Y, Z, G, M, l1, l2, cont)
    T_per = np.zeros(B)

    T_sam_linear = compute_T_linear(X, Y, Z)
    T_per_linear = np.zeros(B)

    def perm_Y(Y, s):
        '''Permutate Y within each bin'''
        new_Y = Y.copy()
        for m in range(M):
            m_ind = list(np.where(G == m)[0])
            m_ind_new = m_ind.copy()
            np.random.seed(s); np.random.shuffle(m_ind_new)
            new_Y[m_ind] = new_Y[m_ind_new]
        return new_Y

    for b in range(B):
        new_Y = perm_Y(Y, b)
        T_per[b] = compute_T(X, new_Y, Z, G, M, l1, l2, cont)
        T_per_linear[b] = compute_T_linear(X, new_Y, Z)
    p = (T_per >= T_sam).sum() / B
    p_linear = (T_per_linear >= T_sam_linear).sum() / B
    return int(p <= alpha), int(p_linear <= alpha)


def data_generative1(N=100, s=1, theta=1):
    '''Generate H0 samples with continuous Z'''
    np.random.seed(s); Z = np.random.uniform(0, 10, N)

    def fz(Z, theta=1):
        return np.array([np.exp(math.sin(theta * z) - np.log(4)) for z in Z])
    
    np.random.seed(s); X = np.random.binomial(1, fz(Z, theta))
    np.random.seed(s+200); Y = np.random.binomial(1, fz(Z ,theta))
    return X, Y, Z

def data_generative2(N=100, s=1):
    '''Generate H1 samples with continuous Z'''
    np.random.seed(s); Z = np.random.uniform(0, 10, N)

    def fz(Z):
        return np.array([np.exp(math.sin(z) - np.log(4)) for z in Z])
    
    X = np.zeros(N)
    Y = np.zeros(N)
    fZ = fz(Z)
    for i in range(N):
        z = fZ[i]
        prob = [z**2 + z/5, (1-z)**2 + z/5, 4*z/5 - z**2, 0]
        prob[3] = 1 - np.sum(prob)
        assert all([x>=0 for x in prob])
        np.random.seed(s)
        situ = np.random.multinomial(1, prob) @ [1, 2, 3, 4]
        if situ == 1:
            X[i] = 1
            Y[i] = 1
        if situ == 2:
            X[i] = 0
            Y[i] = 0
        if situ == 3:
            X[i] = 1
            Y[i] = 0
        if situ == 4:
            X[i] = 0
            Y[i] = 1
    return X, Y, Z

def data_generative3(N=100, s=1, theta=1):
    '''Generate H0 samples with discrete Z'''
    np.random.seed(s+1000); Z = np.random.randint(0, 10, N)

    def fz(Z, theta=1):
        return np.array([np.exp(math.sin(theta * z) - np.log(4)) for z in Z])
    
    np.random.seed(s+1000); X = np.random.binomial(1, fz(Z, theta))
    np.random.seed(s+500); Y = np.random.binomial(1, fz(Z ,theta))
    return X, Y, Z

def experiment1(i, N=100, M=10, theta=1):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative1(N=N, s=i, theta=theta)
    G = compute_G(Z, M=M)
    r1, r2 = LPT(X, Y, Z, G, B = 40, M = M, l1 = 2, l2 = 2, alpha = 0.05)
    return r1, r2

def experiment2(i, N=100, M=10):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative2(N=N, s=i)
    G = compute_G(Z, M=M)
    r1, r2 = LPT(X, Y, Z, G, B = 40, M = M, l1 = 2, l2 = 2, alpha = 0.05)
    return r1, r2

def experiment3(i, N=100, M=10, theta=1):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative3(N=N, s=i, theta=theta)
    G = Z
    r1, r2 = LPT(X, Y, Z, G, B = 40, M = M, l1 = 2, l2 = 2, alpha = 0.05, cont=False)
    return r1, r2

def experiment4(i, N=100, M=10):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative2(N=N, s=i)
    G = np.array([int(z) for z in Z])
    r1, r2 = LPT(X, Y, Z, G, B = 40, M = M, l1 = 2, l2 = 2, alpha = 0.05, cont=False)
    return r1, r2