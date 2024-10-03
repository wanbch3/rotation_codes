import pandas as pd
import numpy as np
from numpy import linalg as la
import cvxpy as cp
import scipy
from scipy.stats import ortho_group
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import time

import os
current_file_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_file_directory)


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


def quadratic_utility_func(x, params): 
    B, D = params
    return -0.5 * cp.quad_form(x, D) + B.T @ x


def customer_step(x, y, params, utility_func):
    M = 10000
    n_product = len(x)
    q = cp.Variable(n_product)
    objective = cp.Maximize(utility_func(x + q, params) - utility_func(x, params))
    constraints = [0 <= q, q <= M*y]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
#     print(q.value)
#     print(result)
    return q.value


def obj_func_general(x, param):
    B1,B2,B3,B4,B5,B6,gamma1,gamma2,gamma3,gamma4,gamma5,gamma6,d01,d02,d03,d04,d05,d15,d23,d24,d34 = x
    B = np.array([B1,B2,B3,B4,B5,B6])
    gamma = np.array([gamma1,gamma2,gamma3,gamma4,gamma5,gamma6])
    D = np.zeros([6,6])
    for i in range(6):
        D[i,i] = 1
    D[0,1] = d01
    D[0,2] = d02
    D[0,3] = d03
    D[0,4] = d04
    D[0,5] = d05
    D[1,5] = d15
    D[2,3] = d23
    D[2,4] = d24
    D[3,4] = d34
    D = D + D.T - np.diag(D.diagonal())
    D = nearestPD(D)

    consumption = param

#     n_products = 6
    x0 = np.zeros(6)
    SSE = 0
    for i in range(len(consumption) - 1):
        row = consumption.iloc[i].to_numpy()
        y = (row != 0).astype(int)
        q = customer_step(x0,y,[B, D],quadratic_utility_func)
        SSE += np.sum((q-row)**2)

        x0 = gamma * (x0 + q)

    row = consumption.iloc[-1].to_numpy()
    y = (row != 0).astype(int)
    q = customer_step(x0,y,[B, D],quadratic_utility_func)
    SSE += np.sum((q-row)**2)
    # print(SSE)
    return SSE


def run_DE_sequential(k,df):
    weekly_scores = df
    mode_list = weekly_scores.columns
    D = np.zeros([len(mode_list),len(mode_list)])
    for i in range(len(mode_list)):
        D[i,i] = 1
    B = np.zeros(len(mode_list))
    gamma = np.zeros(len(mode_list))
#     x_init = np.zeros(len(mode_list))
    bounds = [(0, 100)] * 6 + [(0, 1)] * 6 + [(-1, 1)] * 9
    x00 = [50] * 6 + [0.5] * 6 + [0] * 9
    df = weekly_scores
#     print(df)
    for i in range(k):
        print('start iteration',i)
        results = differential_evolution(func=obj_func_general, bounds=bounds, args=(df,), seed=42, x0=x00)
        x00 = results.x
    B[0],B[1],B[2],B[3],B[4],B[5],gamma[0],gamma[1],gamma[2],gamma[3],gamma[4],gamma[5],D[0,1],D[0,2],D[0,3],D[0,4],D[0,5],D[1,5],D[2,3],D[2,4],D[3,4] = results.x
    D = D + D.T - np.diag(D.diagonal())
    return B,gamma,D
        
if __name__ == "__main__":
    weekly_scores = pd.read_csv('weekly.csv')
    weekly_scores.set_index('date', inplace=True)

    start_time = time.time()

    B,gamma,D = run_DE_sequential(3,weekly_scores)
    np.savetxt('D.txt', D, delimiter=',')
    np.savetxt('B.txt', B, delimiter=',')
    np.savetxt('gamma.txt', gamma, delimiter=',')

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")