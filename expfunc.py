import numpy as np
import pandas as pd
import math
import simufunc
import statsmodels.stats.multitest as ssm
import scipy.stats as st


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
    G = simufunc.compute_G(Z, M=M)
    r1, r2 = simufunc.LPT(X, Y, Z, G, B = 40, M = M, l1 = 2, l2 = 2, alpha = 0.05)
    return r1, r2

def experiment2(i, N=100, M=10):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative2(N=N, s=i)
    G = simufunc.compute_G(Z, M=M)
    r1, r2 = simufunc.LPT(X, Y, Z, G, B = 40, M = M, l1 = 2, l2 = 2, alpha = 0.05)
    return r1, r2

def experiment3(i, N=100, M=10, theta=1):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative3(N=N, s=i, theta=theta)
    G = Z
    r1, r2 = simufunc.LPT(X, Y, Z, G, B = 40, M = M, l1 = 2, l2 = 2, alpha = 0.05, cont=False)
    return r1, r2

def experiment4(i, N=100, M=10):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative2(N=N, s=i)
    G = np.array([int(z) for z in Z])
    r1, r2 = simufunc.LPT(X, Y, Z, G, B = 40, M = M, l1 = 2, l2 = 2, alpha = 0.05, cont=False)
    return r1, r2

def data_generative4(N=100, s=1, theta=1, ss=1):
    '''Generate H0 samples with continuous Z'''
    np.random.seed(s); Z = np.random.uniform(0, 10, N)

    def fz(Z, theta=1):
        return np.array([np.exp(math.sin(theta * z) - np.log(4)) for z in Z])
    
    np.random.seed(s + ss*1000); X = np.random.binomial(1, fz(Z, theta))
    np.random.seed(s + ss*2000); Y = np.random.binomial(1, fz(Z ,theta))
    return X, Y, Z


def experiment5(i, Mt=10, N=100, M=10, theta=1):
    if i%5 == 0:
        print(i)
    def experiment55(ss=1, N=N, s=i, theta=theta, M=M, Mt=Mt):
        X, Y, Z = data_generative4(N=N, s=s, theta=theta, ss=ss)
        G = simufunc.compute_G(Z, M=M)
        r1, r2 = simufunc.LPT(X, Y, Z, G, B = 40, M = M, l1 = 2, l2 = 2, alpha = 0.05/Mt)
        return r1, r2
    res = np.zeros([2, Mt])
    for t in range(Mt):
        res[:, t] = experiment55(ss=t, N=N, s=i, theta=theta, M=M)
    r1 = int(any(ssm.multipletests(res[0,:], method="fdr_bh")[0]))
    r2 = int(any(ssm.multipletests(res[1,:], method="fdr_bh")[0]))

    return r1, r2

def data_generative5(N=100, s=1, type="normal"):
    '''Generate H0 samples with continuous Z'''
    np.random.seed(s); Z = np.random.uniform(0, 10, N)

    if type == "normal":
        np.random.seed(s + N*5); X = np.random.normal(loc=Z, scale=1, size=N)
        np.random.seed(s + N*10); Y = np.random.normal(loc=Z, scale=1, size=N)
    if type == "normal_large":
        np.random.seed(s + N); X = np.random.normal(loc=Z, scale=10, size=N)
        np.random.seed(s + N*2); Y = np.random.normal(loc=Z, scale=10, size=N)
    if type == "normal_small":
        np.random.seed(s + N); X = np.random.normal(loc=Z, scale=.1, size=N)
        np.random.seed(s + N*2); Y = np.random.normal(loc=Z, scale=.1, size=N)
    if type == "chi":
        np.random.seed(s + N); X = np.random.chisquare(df=np.floor(Z)+1, size=N)
        np.random.seed(s + N*2); Y = np.random.chisquare(df=np.floor(Z)+1, size=N)
    if type == "t":
        np.random.seed(s + N); X = np.random.standard_t(df=1, size=N) + Z
        np.random.seed(s + N*2); Y = np.random.standard_t(df=1, size=N) + Z
    if type == "exp":
        np.random.seed(s + N); X = np.random.exponential(scale=Z, size=N)
        np.random.seed(s + N*2); Y = np.random.exponential(scale=Z, size=N)
    if type == "uni":
        np.random.seed(s + N); X = np.random.uniform(low=Z-1, high=Z+1, size=N)
        np.random.seed(s + N*2); Y = np.random.uniform(low=Z-1, high=Z+1, size=N)
    if type == "poi":
        np.random.seed(s + N); X = np.random.poisson(lam=Z, size=N)
        np.random.seed(s + N*2); Y = np.random.poisson(lam=Z, size=N)
    if type == "skewed_normal":
        X = st.skewnorm.rvs(a=-5, loc=Z, scale=2, size=N, random_state=s+N*5)
        Y = st.skewnorm.rvs(a=-5, loc=Z, scale=2, size=N, random_state=s+N*10)
    if type == "skewed_t":
        mean = Z  # Mean
        std_dev = 2  # Standard deviation
        skewness = 5  # Skewness parameter
        df = 5  # Degrees of freedom
        t_samples = st.t.rvs(5, loc=Z, scale=2, size=N, random_state=s+N*5)
        X = t_samples + skewness * np.sqrt((df + 1) / df) * (t_samples - mean) / std_dev
        t_samples = st.t.rvs(5, loc=Z, scale=2, size=N, random_state=s+N*10)
        Y = t_samples + skewness * np.sqrt((df + 1) / df) * (t_samples - mean) / std_dev
        
    return X, Y, Z

def data_generative6(N=100, s=1, type="normal"):
    '''Generate H1 samples with continuous Z'''
    np.random.seed(s); Z = np.random.uniform(0, 10, N)

    if type == "normal":
        data = np.array([st.multivariate_normal.rvs(mean=[z, z], cov=[[1, 0.5],[0.5, 1]], size=1) for z in Z])
        X = data[:, 0]
        Y = data[:, 1]
    if type == "normal_large":
        data = np.array([st.multivariate_normal.rvs(mean=[z, z], cov=[[10, 5],[5, 10]], size=1) for z in Z])
        X = data[:, 0]
        Y = data[:, 1]
    if type == "normal_small":
        data = np.array([st.multivariate_normal.rvs(mean=[z, z], cov=[[0.1, 0.05],[0.05, 0.1]], size=1) for z in Z])
        X = data[:, 0]
        Y = data[:, 1]
    # if type == "uni":
    #     np.random.seed(s); data = np.random.uniform(low=Z-1, high=Z+1, size=(2, N)).T
    #     X = data[:, 0]
    #     Y = data[:, 1]
    if type == "skewed_normal":
        skewness = [5, -5]  # Skewness vector
        normal_samples = np.array([st.multivariate_normal.rvs(mean=[z, z], cov=[[1, 0.5],[0.5, 1]], size=1) for z in Z])
        skew_samples = st.skewnorm.rvs(skewness, loc=0, scale=1, size=(N, 2))
        skewed_normal_samples = normal_samples + skew_samples
        X = skewed_normal_samples[:, 0]
        Y = skewed_normal_samples[:, 1]
    # if type == "skewed_t":
    #     mean = Z  # Mean
    #     cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix
    #     skewness = 5  # Skewness parameter
    #     df = 5  # Degrees of freedom
    #     np.random.seed(s); mv_t_samples = st.multivariate_t.rvs(df, loc=mean, scale=cov, size=1000)
    #     mv_t_samples + np.outer(np.sqrt((df + 1) / df) * skewness, np.linalg.cholesky(cov))
    return X, Y, Z

def experiment6(i, N=100, M=10, type="normal"):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative5(N=N, s=i, type=type)
    G = np.array([int(z) for z in Z])
    p1, p2, p3, p4 = simufunc.LPT(X, Y, Z, G, B = 40, M = M, cont_z=True, cont_xy=True)
    alpha = 0.05
    return int(p1 <= alpha), int(p2 <= alpha), int(p3 <= alpha), int(p4 <= alpha)

def experiment7(i, N=100, M=10, type="normal"):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative6(N=N, s=i, type=type)
    G = np.array([int(z) for z in Z])
    p1, p2, p3, p4 = simufunc.LPT(X, Y, Z, G, B = 40, M = M, cont_z=True, cont_xy=True)
    alpha = 0.05
    return int(p1 <= alpha), int(p2 <= alpha), int(p3 <= alpha), int(p4 <= alpha)

def data_generative7(N=100, s=1, Nvar=8):
    '''Markov chain'''
    assert Nvar >= 3
    X = np.zeros([N, Nvar])
    np.random.seed(s); X[:, 0] = np.random.uniform(0, 10, N)
    for v in range(Nvar):
        if v > 0:
            X[:, v] = np.random.normal(loc=X[:, v-1], scale=1, size=N)
    return X

def experiment8(i, N=100, M=10, Nvar=8):
    if i%5 == 0:
        print(i)
    X = data_generative7(N=N, s=i*10, Nvar=Nvar)
    result1 = np.zeros([Nvar, Nvar])
    result2 = np.zeros([Nvar, Nvar])
    result3 = np.zeros([Nvar, Nvar])
    result4 = np.zeros([Nvar, Nvar])
    alpha = 0.05
    for x,y in combinations(np.arange(Nvar), 2):
        temp_X = X[:, x]
        temp_Y = X[:, y]
        temp_Z = X[:, [np.logical_and(z!=x, z!=y) for z in range(Nvar)]]
        G = simufunc.compute_G(temp_Z)
        p1, p2, p3, p4 = simufunc.LPT(temp_X, temp_Y, temp_Z, G, B = 40, M = M, cont_z=True, cont_xy=True)
        result1[x, y] = int(p1 <= alpha)
        result1[y, x] = int(p1 <= alpha)
        result2[x, y] = int(p2 <= alpha)
        result2[y, x] = int(p2 <= alpha)
        result3[x, y] = int(p3 <= alpha)
        result3[y, x] = int(p3 <= alpha)
        result4[x, y] = int(p4 <= alpha)
        result4[y, x] = int(p4 <= alpha)

    return result1, result2, result3, result4

def experiment9(i, N=100, M=10, type="normal", sub=0):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative8(N=N, s=i, type=type, hypo="h1")
    G = np.array([int(z) for z in Z])
    p1, p2, p3, p4, p5, p6, p7, p8 = simufunc.LPT(X, Y, Z, G, B = 40, M = M, cont_z=True, cont_xy=True, sub=sub)
    alpha = 0.05
    return int(p1 <= alpha), int(p2 <= alpha), int(p3 <= alpha), int(p4 <= alpha),\
           int(p5 <= alpha), int(p6 <= alpha), int(p7 <= alpha), int(p8 <= alpha)

def experiment99(i, N=100, M=10, type="normal", sub=0):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative8(N=N, s=i, type=type, hypo="h0")
    G = np.array([int(z) for z in Z])
    p1, p2, p3, p4, p5, p6, p7, p8 = LPT(X, Y, Z, G, B = 40, M = M, cont_z=True, cont_xy=True, sub=sub)
    alpha = 0.05
    return int(p1 <= alpha), int(p2 <= alpha), int(p3 <= alpha), int(p4 <= alpha),\
           int(p5 <= alpha), int(p6 <= alpha), int(p7 <= alpha), int(p8 <= alpha)




def Z_to_Y(Z):
    return Z+3*(Z**2)+2

def data_generative8(N=100, s=1, type="normal", hypo="h0", xfun=None, yfun=None):
    '''Generate H0 samples with continuous Z'''
    np.random.seed(s); Z = np.random.uniform(0, 10, N)

    if xfun == None:
        Zx = Z
    else:
        Zx = xfun(Z)

    if yfun == None:
        Zy = Z
    else:
        Zy = yfun(Z)   

    if hypo == "h0":
        if type == "normal":
            np.random.seed(s + N*5); X = np.random.normal(loc=Zx, scale=1, size=N)
            np.random.seed(s + N*10); Y = np.random.normal(loc=Zy, scale=1, size=N)
        elif type == "normal_large":
            np.random.seed(s + N*5); X = np.random.normal(loc=Zx, scale=10, size=N)
            np.random.seed(s + N*10); Y = np.random.normal(loc=Zy, scale=10, size=N)
        elif type == "normal_small":
            np.random.seed(s + N*5); X = np.random.normal(loc=Zx, scale=.1, size=N)
            np.random.seed(s + N*10); Y = np.random.normal(loc=Zy, scale=.1, size=N)
        elif type == "chi":
            np.random.seed(s + N*5); X = np.random.chisquare(df=5, size=N) + Zx
            np.random.seed(s + N*10); Y = np.random.chisquare(df=5, size=N) + Zy
        elif type == "t":
            np.random.seed(s + N*5); X = np.random.standard_t(df=1, size=N) + Zx
            np.random.seed(s + N*10); Y = np.random.standard_t(df=1, size=N) + Zy
        elif type == "exp":
            np.random.seed(s + N*5); X = np.random.exponential(scale=1, size=N) + Zx
            np.random.seed(s + N*10); Y = np.random.exponential(scale=1, size=N) + Zy
        elif type == "uni":
            np.random.seed(s + N*5); X = np.random.uniform(low=Zx-1, high=Zx+1, size=N)
            np.random.seed(s + N*10); Y = np.random.uniform(low=Zy-1, high=Zy+1, size=N)
        elif type == "poi":
            np.random.seed(s + N*5); X = np.random.poisson(lam=2, size=N) + Zx
            np.random.seed(s + N*10); Y = np.random.poisson(lam=2, size=N) + Zy
        elif type == "skewed_normal":
            X = st.skewnorm.rvs(a=-5, loc=Zx, scale=2, size=N, random_state=s+N*5)
            Y = st.skewnorm.rvs(a=-5, loc=Zy, scale=2, size=N, random_state=s+N*10)
        elif type == "skewed_t":
            mean = Z  # Mean
            std_dev = 2  # Standard deviation
            skewness = 5  # Skewness parameter
            df = 5  # Degrees of freedom
            t_samples = st.t.rvs(5, loc=Zx, scale=2, size=N, random_state=s+N*5)
            X = t_samples + skewness * np.sqrt((df + 1) / df) * (t_samples - mean) / std_dev
            t_samples = st.t.rvs(5, loc=Zy, scale=2, size=N, random_state=s+N*10)
            Y = t_samples + skewness * np.sqrt((df + 1) / df) * (t_samples - mean) / std_dev
        else:
            raise ValueError("Non-existing distribution type!")
    
    elif hypo == "h1":
        Zxy = np.column_stack((Zx, Zy))
        if type == "normal":
            data = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[1, 0.5],[0.5, 1]], size=1) for i in range(Zxy.shape[0])])
            X = data[:, 0]
            Y = data[:, 1]
        elif type == "normal_large":
            data = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[10, 5],[5, 10]], size=1) for i in range(Zxy.shape[0])])
            X = data[:, 0]
            Y = data[:, 1]
        elif type == "normal_small":
            data = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[0.1, 0.05],[0.05, 0.1]], size=1) for i in range(Zxy.shape[0])])
            X = data[:, 0]
            Y = data[:, 1]
        elif type == "uni":
            np.random.seed(s); data = np.random.uniform(low=Zxy-1, high=Zxy+1, size=(N, 2))
            X = data[:, 0]
            Y = data[:, 1]
        elif type == "skewed_normal":
            skewness = [5, -5]  # Skewness vector
            normal_samples = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[1, 0.5],[0.5, 1]], size=1) for i in range(Zxy.shape[0])])
            skew_samples = st.skewnorm.rvs(skewness, loc=0, scale=1, size=(N, 2))
            skewed_normal_samples = normal_samples + skew_samples
            X = skewed_normal_samples[:, 0]
            Y = skewed_normal_samples[:, 1]
        else:
            raise ValueError("Non-existing distribution type!")
    else:
        raise ValueError("Non-existing Hypothesis type!")
    # if type == "skewed_t":
    #     mean = Z  # Mean
    #     cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix
    #     skewness = 5  # Skewness parameter
    #     df = 5  # Degrees of freedom
    #     np.random.seed(s); mv_t_samples = st.multivariate_t.rvs(df, loc=mean, scale=cov, size=1000)
    #     mv_t_samples + np.outer(np.sqrt((df + 1) / df) * skewness, np.linalg.cholesky(cov))
    return X, Y, Z

def experiment10(i, N=100, M=10, type="normal", sub=0):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative8(N=N, s=i, type=type, hypo="h0", yfun=Z_to_Y)
    G = np.array([int(z) for z in Z])
    p1, p2, p3, p4, p5, p6, p7, p8 = simufunc.LPT(X, Y, Z, G, B = 40, M = M, cont_z=True, cont_xy=True, sub=sub)
    alpha = 0.05
    return int(p1 <= alpha), int(p2 <= alpha), int(p3 <= alpha), int(p4 <= alpha),\
           int(p5 <= alpha), int(p6 <= alpha), int(p7 <= alpha), int(p8 <= alpha)

def experiment11(i, N=100, M=10, type="normal", sub=0):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative8(N=N, s=i, type=type, hypo="h1", yfun=Z_to_Y)
    G = np.array([int(z) for z in Z])
    p1, p2, p3, p4, p5, p6, p7, p8 = simufunc.LPT(X, Y, Z, G, B = 40, M = M, cont_z=True, cont_xy=True, sub=sub)
    alpha = 0.05
    return int(p1 <= alpha), int(p2 <= alpha), int(p3 <= alpha), int(p4 <= alpha),\
           int(p5 <= alpha), int(p6 <= alpha), int(p7 <= alpha), int(p8 <= alpha)

def Z_to_Y2(Z):
    return np.log(Z+1)-2

def Z_to_Y3(Z):
    return 5*Z

def Z_to_Y4(Z):
    return np.log(Z+1)+2

def Z_to_Y5(Z):
    return Z+2*Z**2+Z**3

def Z_to_Y6(Z):
    return 5/Z

def Z_to_Y7(Z):
    return 7+Z**(1/2)

def Z_to_Y8(Z):
    return -Z


def experiment12(i, N=100, M=10, type="normal", sub=0, hypo="h1", xfun=None, yfun=None):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative8(N=N, s=i, type=type, hypo=hypo, yfun=yfun, xfun=xfun)
    G = np.array([int(z) for z in Z])
    p1, p2, p3, p4, p5, p6, p7, p8 = simufunc.LPT(X, Y, Z, G, B = 50, M = M, cont_z=True, cont_xy=True, sub=sub)
    alpha = 0.05
    return int(p1 <= alpha), int(p2 <= alpha), int(p3 <= alpha), int(p4 <= alpha),\
           int(p5 <= alpha), int(p6 <= alpha), int(p7 <= alpha), int(p8 <= alpha)


def data_generative9(N=100, s=1, type="normal", hypo="h0", xfun=None, yfun=None):
    '''Generate H0 samples with continuous Z'''
    np.random.seed(s); Z = np.random.uniform(0, 10, N)

    if xfun == None:
        Zx = Z
    else:
        Zx = xfun(Z)

    if yfun == None:
        Zy = Z
    else:
        Zy = yfun(Z)   

    if hypo == "h0":
        if type == "normal":
            np.random.seed(s + N*50); X = np.random.normal(loc=Zx, scale=5, size=N)
            np.random.seed(s + N*10); Y = np.random.normal(loc=Zy, scale=.5, size=N)
        elif type == "uni":
            np.random.seed(s + N*50); X = np.random.uniform(low=Zx-1, high=Zx+1, size=N)
            np.random.seed(s + N*10); Y = np.random.uniform(low=Zy-.2, high=Zy+.2, size=N)
        elif type == "poi":
            np.random.seed(s + N*50); X = np.random.poisson(lam=2, size=N) + Zx
            np.random.seed(s + N*10); Y = np.random.poisson(lam=1, size=N) + Zy
        elif type == "skewed_normal":
            X = st.skewnorm.rvs(a=-5, loc=Zx, scale=1, size=N, random_state=s+N*50)
            Y = st.skewnorm.rvs(a=-5, loc=Zy, scale=.1, size=N, random_state=s+N*10)
        else:
            raise ValueError("Non-existing distribution type!")
    
    elif hypo == "h1":
        Zxy = np.column_stack((Zx, Zy))
        if type == "normal":
            data = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[5, 1.2],[1.2, .5]], size=1) for i in range(Zxy.shape[0])])
            X = data[:, 0]
            Y = data[:, 1]
        elif type == "skewed_normal":
            skewness = [5, -5]  # Skewness vector
            normal_samples = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[4, 0.15],[0.15, 0.01]], size=1) for i in range(Zxy.shape[0])])
            skew_samples = st.skewnorm.rvs(skewness, loc=0, scale=[1, 0.01], size=(N, 2))
            skewed_normal_samples = normal_samples + skew_samples
            X = skewed_normal_samples[:, 0]
            Y = skewed_normal_samples[:, 1]
        elif type == "skewed_normal_large":
            skewness = [5, -5]  # Skewness vector
            normal_samples = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[4, 1.2],[1.2, 0.5]], size=1) for i in range(Zxy.shape[0])])
            skew_samples = st.skewnorm.rvs(skewness, loc=0, scale=[2, 0.1], size=(N, 2))
            skewed_normal_samples = normal_samples + skew_samples
            X = skewed_normal_samples[:, 0]
            Y = skewed_normal_samples[:, 1]
        else:
            raise ValueError("Non-existing distribution type!")
    else:
        raise ValueError("Non-existing Hypothesis type!")
    # if type == "skewed_t":
    #     mean = Z  # Mean
    #     cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix
    #     skewness = 5  # Skewness parameter
    #     df = 5  # Degrees of freedom
    #     np.random.seed(s); mv_t_samples = st.multivariate_t.rvs(df, loc=mean, scale=cov, size=1000)
    #     mv_t_samples + np.outer(np.sqrt((df + 1) / df) * skewness, np.linalg.cholesky(cov))
    return X, Y, Z



def experiment13(i, N=100, M=10, type="normal", sub=0, hypo="h1", xfun=None, yfun=None):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative9(N=N, s=i, type=type, hypo=hypo, yfun=yfun, xfun=xfun)
    G = simufunc.compute_G(Z)
    p1, p2, p3, p4, p5, p6, p7, p8 = simufunc.LPT(X, Y, Z, G, B = 40, M = M, cont_z=True, cont_xy=True, sub=sub)
    alpha = 0.05
    return int(p1 <= alpha), int(p2 <= alpha), int(p3 <= alpha), int(p4 <= alpha),\
           int(p5 <= alpha), int(p6 <= alpha), int(p7 <= alpha), int(p8 <= alpha)

def data_generative10(N=100, s=1, type="normal", hypo="h0", xfun=None, yfun=None):
    '''Generate H0 samples with continuous Z'''
    np.random.seed(s); Z = np.random.uniform(0, 10, N)

    if xfun == None:
        Zx = Z
    else:
        Zx = xfun(Z)

    if yfun == None:
        Zy = Z
    else:
        Zy = yfun(Z)   

    if hypo == "h0":
        if type == "normal":
            np.random.seed(s + N*1); X = np.random.normal(loc=Zx, scale=.1, size=N)
            np.random.seed(s + N*2); Y = np.random.normal(loc=Zy, scale=.1, size=N)
        elif type == "uni":
            np.random.seed(s + N*1); X = np.random.uniform(low=Zx-.2, high=Zx+.2, size=N)
            np.random.seed(s + N*2); Y = np.random.uniform(low=Zy-.2, high=Zy+.2, size=N)
        elif type == "poi":
            np.random.seed(s + N*1); X = np.random.poisson(lam=1, size=N) + Zx
            np.random.seed(s + N*2); Y = np.random.poisson(lam=1, size=N) + Zy
        elif type == "skewed_normal":
            X = st.skewnorm.rvs(a=-5, loc=Zx, scale=.1, size=N, random_state=s+N*1)
            Y = st.skewnorm.rvs(a=-5, loc=Zy, scale=.1, size=N, random_state=s+N*10)
        else:
            raise ValueError("Non-existing distribution type!")
    
    elif hypo == "h1":
        Zxy = np.column_stack((Zx, Zy))
        if type == "normal":
            data = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[0.1, 0.07],[0.07, 0.1]], size=1) for i in range(Zxy.shape[0])])
            X = data[:, 0]
            Y = data[:, 1]
        elif type == "skewed_normal":
            skewness = [5, -5]  # Skewness vector
            normal_samples = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[0.1, 0.07],[0.07, 0.1]], size=1) for i in range(Zxy.shape[0])])
            skew_samples = st.skewnorm.rvs(skewness, loc=0, scale=[0.1, 0.1], size=(N, 2))
            skewed_normal_samples = normal_samples + skew_samples
            X = skewed_normal_samples[:, 0]
            Y = skewed_normal_samples[:, 1]
        else:
            raise ValueError("Non-existing distribution type!")
    else:
        raise ValueError("Non-existing Hypothesis type!")
    # if type == "skewed_t":
    #     mean = Z  # Mean
    #     cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix
    #     skewness = 5  # Skewness parameter
    #     df = 5  # Degrees of freedom
    #     np.random.seed(s); mv_t_samples = st.multivariate_t.rvs(df, loc=mean, scale=cov, size=1000)
    #     mv_t_samples + np.outer(np.sqrt((df + 1) / df) * skewness, np.linalg.cholesky(cov))
    return X, Y, Z



def experiment14(i, N=100, M=10, type="normal", sub=0, hypo="h1", xfun=None, yfun=None):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative10(N=N, s=i, type=type, hypo=hypo, yfun=yfun, xfun=xfun)
    G = simufunc.compute_G(Z)
    p1, p2, p3, p4, p5, p6, p7, p8 = simufunc.LPT(X, Y, Z, G, B = 40, M = M, cont_z=True, cont_xy=True, sub=sub)
    alpha = 0.05
    return int(p1 <= alpha), int(p2 <= alpha), int(p3 <= alpha), int(p4 <= alpha),\
           int(p5 <= alpha), int(p6 <= alpha), int(p7 <= alpha), int(p8 <= alpha)


def data_generative11(N=100, s=1, type="normal", hypo="h0", xfun=None, yfun=None):
    '''Generate H0 samples with continuous Z'''
    np.random.seed(s); Z = np.random.uniform(0, 10, N)

    if xfun == None:
        Zx = Z
    else:
        Zx = xfun(Z)

    if yfun == None:
        Zy = Z
    else:
        Zy = yfun(Z)   

    if hypo == "h0":
        if type == "normal":
            np.random.seed(s + N*50); X = np.random.normal(loc=Zx, scale=1, size=N)
            np.random.seed(s + N*10); Y = np.random.normal(loc=Zy, scale=1, size=N)
        elif type == "uni":
            np.random.seed(s + N*50); X = np.random.uniform(low=Zx-1, high=Zx+1, size=N)
            np.random.seed(s + N*10); Y = np.random.uniform(low=Zy-1, high=Zy+1, size=N)
        elif type == "poi":
            np.random.seed(s + N*50); X = np.random.poisson(lam=2, size=N) + Zx
            np.random.seed(s + N*10); Y = np.random.poisson(lam=2, size=N) + Zy
        elif type == "skewed_normal":
            X = st.skewnorm.rvs(a=-5, loc=Zx, scale=1, size=N, random_state=s+N*50)
            Y = st.skewnorm.rvs(a=-5, loc=Zy, scale=1, size=N, random_state=s+N*10)
        else:
            raise ValueError("Non-existing distribution type!")
    
    elif hypo == "h1":
        Zxy = np.column_stack((Zx, Zy))
        if type == "normal":
            data = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[1, 0.7],[0.7, 1]], size=1) for i in range(Zxy.shape[0])])
            X = data[:, 0]
            Y = data[:, 1]
        elif type == "skewed_normal":
            skewness = [5, -5]  # Skewness vector
            normal_samples = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[1, 0.7],[0.7, 1]], size=1) for i in range(Zxy.shape[0])])
            skew_samples = st.skewnorm.rvs(skewness, loc=0, scale=[1, 1], size=(N, 2))
            skewed_normal_samples = normal_samples + skew_samples
            X = skewed_normal_samples[:, 0]
            Y = skewed_normal_samples[:, 1]
        else:
            raise ValueError("Non-existing distribution type!")
    else:
        raise ValueError("Non-existing Hypothesis type!")
    # if type == "skewed_t":
    #     mean = Z  # Mean
    #     cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix
    #     skewness = 5  # Skewness parameter
    #     df = 5  # Degrees of freedom
    #     np.random.seed(s); mv_t_samples = st.multivariate_t.rvs(df, loc=mean, scale=cov, size=1000)
    #     mv_t_samples + np.outer(np.sqrt((df + 1) / df) * skewness, np.linalg.cholesky(cov))
    return X, Y, Z



def experiment15(i, N=100, M=10, type="normal", sub=0, hypo="h1", xfun=None, yfun=None):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative11(N=N, s=i, type=type, hypo=hypo, yfun=yfun, xfun=xfun)
    G = simufunc.compute_G(Z)
    p1, p2, p3, p4, p5, p6, p7, p8 = simufunc.LPT(X, Y, Z, G, B = 40, M = M, cont_z=True, cont_xy=True, sub=sub)
    alpha = 0.05
    return int(p1 <= alpha), int(p2 <= alpha), int(p3 <= alpha), int(p4 <= alpha),\
           int(p5 <= alpha), int(p6 <= alpha), int(p7 <= alpha), int(p8 <= alpha)