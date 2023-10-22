import numpy as np
from data import Data
import typing
import random
from copy import deepcopy
from util import init_T

class SOGA:
    def __init__(self):
        pass

    def adapt(self, data: Data):
        """
        Setup for solving a particular network, take in network data
        """
        self.n = data.n
        self.m = data.m
        self.q = data.q
        self.K = data.K
        self.T = init_T(data)
        self.__build()

    def __build(self):
        self.F = [None]*self.n
        for i in range(self.n):
            self.F[i] = list(range(self.q))
        self.N = set(range(self.m))

    def solve(self, mode='linear'):
        """
        Solve for adapted network with
        - mode: 'linear' (ILP), 'quadratic' (IQP), 'prioritize' (pIQP) or 'reduced_variance' (rvIQP)
        Return sensors_mask and achieved coverage for each target via C and a
        """
        F = self.F.copy()
        N = self.N.copy()
        C = [None]*self.n
        a = np.zeros(self.m)
        T = self.T
        K = self.K

        while self.__continue(F, N):
            _, si, pj = self.max_benefit(F=F, T=T, a=a, K=K, mode=mode)
            if si is None or pj is None:
                break
            C[si] = pj
            for t in range(self.m):
                if T[si, t, pj] and t in N:
                    a[t] += 1
                    if a[t] == K[t]:
                        N.remove(t)
            F[si] = None
        
        C = np.array(C)
        a = np.array(a)
        C[C == None] = self.q

        return C, np.array(a)

    def max_benefit(self, F, T, a, K, mode):
        si, pj = None, None
        max_value = -1

        for i in range(self.n):
            if F[i] == None:
                continue
            for j in F[i]:
                value = 0
                flag = False
                for t in range(self.m):
                    if T[i, t, j] and a[t] < K[t]:
                        flag = True
                        if mode == 'linear':
                            value += 1
                        elif mode == 'quadratic':
                            value += ((K[t]-a[t])**2 - (K[t]-a[t]-1)**2)
                        elif mode == 'prioritize':
                            value += K[t]*((K[t]-a[t])**2 - (K[t]-a[t]-1)**2)
                        elif mode == 'reduced_variance':
                            count = 0
                            group_achived_coverage = 0
                            for t_i in range(len(K)):
                                if K[t_i] == K[t]:
                                    count += 1
                                    group_achived_coverage += a[t_i]
                            u, g = group_achived_coverage/count, count
                            old = (K[t]-a[t])**2 + (a[t]-u)**2/g
                            new = (K[t]-a[t]-1)**2 + (a[t]-u+1-1/g)**2/g
                            value += old-new
                if flag and value > max_value:
                    max_value = value
                    si = i
                    pj = j

        return value, si, pj          

    def __continue(self, F, N):
        return any(F) and any(N)
