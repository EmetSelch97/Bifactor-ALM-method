# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:05:03 2025

@author: ZZ
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as rgt
import pandas as pd
from scipy.optimize import minimize
#from tqdm import tqdm

from Permutation import BF_permutation,Compare_Q,Average_Cr,Exact_Cr
from multiprocessing import Pool

# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
# from rpy2.robjects.vectors import FloatVector
# from rpy2.robjects import r, globalenv,numpy2ri

#import matplotlib.pyplot as plt
import time

#GPArotation = importr('GPArotation')

class ebfa_solver():
    def __init__(self,S):
        self.S = S
    def efa_decom(self,x,J,C):
        if len(x) != J*(C+1)- int(C*(C-1)/2):
            raise  NotImplementedError
        d = x[J*C-int(C*(C-1)/2) : J*(C+1)-int(C*(C-1)/2)]
        D = np.diag(d**2)
        L = np.zeros([J,C])
        for i in range(C):
            L[i:,i] = x[int(i*J - i*(i-1)/2) : int((i+1)*J - i*(i+1)/2)]

        return L,D,d
    def efa_nll(self,x,*args):
        J,C,S = args
        L,D,d = self.efa_decom(x,J,C)

        Cov = L @ L.T + D
        R = np.linalg.solve(Cov,S)

        loss = np.log(np.linalg.det(2*np.pi*Cov))/2 + np.trace(R)/2
        return loss
    
    def efa_grad(self,x,*args):
        J,C,S = args
        L,D,d = self.efa_decom(x,J,C)
        
        Cov = L @ L.T + D
        inv_Cov = np.linalg.solve(Cov,np.identity(J))

        Sdw =  inv_Cov @ (S @ inv_Cov)

        diff = inv_Cov -Sdw
        
        grad_L = diff @ L
        grad_D = diff/2

        grad_x = np.zeros_like(x)
        grad_x[J*C-int(C*(C-1)/2) : J*(C+1)-int(C*(C-1)/2)] = 2*np.diag(grad_D)*d

        for i in range(C):
            grad_x[int(i*J - i*(i-1)/2) : int((i+1)*J - i*(i+1)/2)] = grad_L[i:,i]
        return grad_x
    
    def efa_init(self,J,C):
        x_init = np.zeros(J*(C+1)- int(C*(C-1)/2))
        x_init[:J*C- int(C*(C-1)/2)] = rgt.randn(J*C- int(C*(C-1)/2))
        x_init[J*C- int(C*(C-1)/2): J*(C+1)- int(C*(C-1)/2)] = 1 + rgt.rand(J)

        return x_init
    
    def stan_trans(self,x,G):
        if len(x) != int(G*(G-1)/2):
            raise NotImplementedError('Dimension of input does not match parameters in the corr matrix')

        h = np.zeros([G,G])
        for i in range(1,G):
            h[0:i,i] = x[int((i-1)*i/2):int((i+1)*i/2)]
        Z = np.tanh(h)

        U = np.zeros([G,G])
        U[0,0] = 1
        U[0,1:G] = Z[0,1:G]
        for i in range(1,G):
            U[i,i] = U[i-1,i]*np.sqrt(1-Z[i-1,i]**2) / Z[i-1,i]
            U[i,(i+1):G] = Z[i,(i+1):G]*U[i-1,(i+1):G]*np.sqrt(1 - Z[i-1,(i+1):G]**2) / Z[i-1,(i+1):G] 
        return U
    
    def cons_fun(self,x,J,G,Pair_cons):
        p = len(Pair_cons)
        if p != int((G-1)*G/2):
            raise NotImplementedError('The dimension of constraint pair does not match bi-factor model ')

        cons_val = np.zeros([p,J])
        for i in range(p):
            Z = Pair_cons[i]
            cons_val[i,:] = x[int(G*(G-1)/2 + Z[0]*J): int(G*(G-1)/2 + (Z[0]+1)*J)] * x[int(G*(G-1)/2 + Z[1]*J): int(G*(G-1)/2 + (Z[1]+1)*J)]
        return cons_val
    def con_gd(self,L,J,G,Pairs,gamma,rho):
        L2 = L**2
        grad_L = np.zeros([J,(G+1)])
        for i in range(len(Pairs)):
            Z = Pairs[i]
            a = Z[0]
            b = Z[1]
            gm = gamma[i,:]
            grad_L[:,a] = grad_L[:,a] + gm*L[:,b] + rho*L[:,a]*L2[:,b]
            grad_L[:,b] = grad_L[:,b] + gm*L[:,a] + rho*L[:,b]*L2[:,a]
        grad = np.zeros(J*(1+G) + int(G*(G-1)/2) + J)
        #for i in range(1+G):
        #    grad[int(G*(G-1)/2 + i*J): int(G*(G-1)/2 + (i+1)*J)] = grad_L[:,i]
        grad[int(G*(G-1)/2):(J*(1+G)+int(G*(G-1)/2))] = grad_L.T.reshape(J*(G+1))
        return grad
    
    def gd_rp(self,x,dU,G):
        if len(x)!=int(G*(G-1)/2):
            raise NotImplementedError
        if len(dU) != int((G-1)*(G+2)/2):
            raise NotImplementedError
        #Z = (np.exp(2*x) - 1) / (np.exp(2*x) + 1) 
        Z = np.tanh(x)
        dZ = 1-Z**2
        A = np.zeros([int(G*(G-1)/2),int((G-1)*(G+2)/2)])

        for i in range(1,G):
            y = Z[int(i*(i-1)/2): int(i*(i+1)/2)]
            sy = np.sqrt(1-y**2)
            inv_sy = 1/(sy + 1e-15)
            #inv_sy = 1/sy
            scaler_full = np.ones_like(y)
            scaler_full[1:] = sy[:-1]
            scaler_full = np.cumprod(scaler_full)

            A_sub = np.zeros([i,i+1])
            for j in range(i):
                A_sub[j,j] = 1
                r_1 = np.ones(i-j)
                r_1[:-1] = y[j+1:]
                r_2 = np.ones(i-j)
                r_2[1:] = sy[j+1:]
                r_2 = np.cumprod(r_2)
                A_sub[j,j+1:] = -r_1*r_2*inv_sy[j]*y[j]
            A_sub = np.diag(scaler_full) @ A_sub
            A[int(i*(i-1)/2): int(i*(i+1)/2), int(i*(i+1)/2-1): int(i*(i+3)/2)] = A_sub   
        gd = (A @ dU)* dZ
        return gd
    
    def alm_gd(self,x,*args):
        J,G,S,gamma,rho,Pairs = args
        if len(x) != J*(1+G) + int(G*(G-1)/2) + J:
            raise NotImplementedError('The dimension of input does not match the parameters in bi-fcator model')

        Phi = np.zeros([(1+G),(1+G)])
        Phi[0,0] = 1
        Phi[1:(G+1),1:(G+1)] = self.stan_trans(x[0:int(G*(G-1)/2)],G)

        L = np.zeros([J,(G+1)])
        for i in range(G+1):
            L[:,i] = x[int(G*(G-1)/2 + i*J): int(G*(G-1)/2 + (i+1)*J)]
        
        D = np.diag(x[int(G*(G-1)/2 + (G+1)*J): int(G*(G-1)/2 + (G+2)*J)]**2)    
        d = x[int(G*(G-1)/2 + (G+1)*J): int(G*(G-1)/2 + (G+2)*J)]
        
        Psi = Phi.T @ Phi
        Cov = L @ (Psi @ L.T) + D

        inv_Cov = np.linalg.solve(Cov,np.identity(J))

        Sdw = inv_Cov @ (S @ (inv_Cov))
        LP = L @ Psi
        PL = Phi @ L.T

        cons_grad = self.con_gd(L,J,G,Pairs,gamma,rho)
        
        nll_grad = np.zeros_like(x)
        
        dL = inv_Cov @ LP - Sdw @ LP
        #for i in range(G+1):
        #    nll_grad[int(G*(G-1)/2 + i*J): int(G*(G-1)/2 + (i+1)*J)] = dL[:,i]
        nll_grad[int(G*(G-1)/2):int(G*(G-1)/2 + (G+1)*J)] = dL.T.reshape((G+1)*J)
        
        dd = np.diag( inv_Cov - Sdw ) * d
        nll_grad[int(G*(G-1)/2 + (G+1)*J): int(G*(G-1)/2 + (G+2)*J)] = dd

        dPhi = PL @ (inv_Cov @ L) - PL @ (Sdw @ L)
        dU = np.zeros(int((G-1)*(G+2)/2))
        for i in range(G-1):
            dU[int(i*(i+3)/2) : int(i*(i+5)/2 + 2)] = dPhi[1:(i+3),i+2]

        dphi = self.gd_rp(x[0:int(G*(G-1)/2)],dU,G)
        nll_grad[0:int(G*(G-1)/2)] = dphi
        full_grad = nll_grad + cons_grad
        return full_grad
    
    def objective_function(self,x,*args):
        J,G,S,gamma,rho,Pairs = args
        if len(x) != J*(1+G) + int(G*(G-1)/2) + J:
            raise NotImplementedError
        Phi = np.zeros([(1+G),(1+G)])
        Phi[0,0] = 1
        #for i in range(G):
        #    Phi[1:(2+i),(1+i)] = x[int((i+1)*i/2):int((i+3)*i/2+1)]
        Phi[1:(G+1),1:(G+1)] = self.stan_trans(x[0:int(G*(G-1)/2)],G)
        
        L = np.zeros([J,(G+1)])
        for i in range(G+1):
            L[:,i] = x[int(G*(G-1)/2 + i*J): int(G*(G-1)/2 + (i+1)*J)]
        D = np.diag(x[int(G*(G-1)/2 + (G+1)*J): int(G*(G-1)/2 + (G+2)*J)]**2)

        Psi = Phi.T @ Phi
        Cov = L @ (Psi @ L.T) + D
        R = np.linalg.solve(Cov,S)

        loss_part = np.log(np.linalg.det(2*np.pi*Cov))/2 + np.trace(R)/2

        cons_val = self.cons_fun(x,J,G,Pairs)
        pen_part = np.sum(gamma*cons_val) + 0.5*rho*np.sum(cons_val**2)

        loss = loss_part + pen_part
        return loss
    
    def para_decompose(self,x,J,G):
        if len(x) != J*(1+G) + int(G*(G-1)/2) + J:
            raise NotImplementedError('The dimension of x does not match the parameters of bifactor model')
        Phi = np.zeros([(1+G),(1+G)])
        Phi[0,0] = 1
        Phi[1:(G+1),1:(G+1)] = self.stan_trans(x[0:int(G*(G-1)/2)],G)
        #L = np.zeros([J,(G+1)])
        #for i in range(G+1):
        #    L[:,i] = x[int(G*(G-1)/2 + i*J): int(G*(G-1)/2 + (i+1)*J)]
        L = x[int(G*(G-1)/2):int(G*(G-1)/2 + (G+1)*J)].reshape((G+1),J).T
        d = x[int(G*(G-1)/2 + (G+1)*J): int(G*(G-1)/2 + (G+2)*J)]
        D = np.diag(d**2)
        
        Psi = Phi.T @ Phi
        Cov = L @ (Psi @ L.T) + D
        
        return L,D,d,Psi,Cov
    
    # def init_value(self,J,G):
    #     x = np.zeros(J*(1+G) + int(G*(G-1)/2) + J)
    #     x[0:int(G*(G-1)/2)] = rgt.randn(int(G*(G-1)/2))
        
    #     x[J*(1+G) + int(G*(G-1)/2) : J*(1+G) + int(G*(G-1)/2) + J] = 1 + rgt.rand(J)
        
    #     x[int(G*(G-1)/2): int(G*(G-1)/2) + J] = rgt.rand(J)
        
    #     x[int(G*(G-1)/2)+J : int(G*(G-1)/2) + J*(1+G)] = 0.1*rgt.randn(G*J)
        
    #     return x
    
    def init_value(self,J,G):
        x = np.zeros(J*(1+G) + int(G*(G-1)/2) + J)
        x[0:int(G*(G-1)/2)] = rgt.uniform(-0.5,0.5,int(G*(G-1)/2))
        
        x[J*(1+G) + int(G*(G-1)/2) : J*(1+G) + int(G*(G-1)/2) + J] = 1 + rgt.rand(J)
        
        x[int(G*(G-1)/2): int(G*(G-1)/2) + J] = rgt.rand(J)
        
        x[int(G*(G-1)/2)+J : int(G*(G-1)/2) + J*(1+G)] = rgt.uniform(-0.5,0.5,G*J)
        
        return x
    
    def alm_solve(self,x_init,J,G,S,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter):
        if len(x_init) == 0 :
            x_init = self.init_value(J,G)
        p = len(x_init)
        x_old = x_init.copy()
        
        gamma = gamma_0.copy()
        rho = rho_0
           
        L_old,D_old,d_old,Psi_old,Cov_old = self.para_decompose(x_old,J,G)
        cons_val_old = self.cons_fun(x_old,J,G,Pair)
        
        dist_val = np.max(np.sort(np.abs(L_old[:,1:]),axis=1)[:,-2])
        dist_para = np.linalg.norm(x_old)/np.sqrt(p)
        #print(dist_val)
        
        iter_num = 0
        while max(dist_para,dist_val) > tol and iter_num < max_iter:
            result = minimize(self.objective_function,x_old,args=(J,G,S,gamma,rho,Pair),method = 'L-BFGS-B',jac = self.alm_gd)
            x_new = result.x
            dist_para = np.linalg.norm(x_old - x_new)/np.sqrt(p)    
            cons_val_new =  self.cons_fun(x_new,J,G,Pair)
            gamma = gamma + rho *cons_val_new
            if np.linalg.norm(cons_val_old,ord = 'fro') > theta*np.linalg.norm(cons_val_new,ord = 'fro'):
                rho = rho*rho_sigma
                
            x_old = x_new.copy()
            L_old,D_old,d_old,Psi_old,Cov_old = self.para_decompose(x_old,J,G)
            dist_val = np.max(np.sort(np.abs(L_old[:,1:]),axis=1)[:,-2])
            #print(dist_val)
            iter_num = iter_num + 1
        return x_old,iter_num,dist_val
    
    def efa_solve(self,J,G,S,x_init = np.array([])):
        C=G+1
        if len(x_init) == 0:
            x_init = self.efa_init(self,J,C)
        result = minimize(self.efa_nll,x_init,args=(J,C,S),method = 'L-BFGS-B',jac = self.efa_grad)
        L_est,D_est,d_est = self.efa_decom(result.x,J,C)
        return L_est,D_est,d_est
    
    
def stan_trans(x,G):
    if len(x) != int(G*(G-1)/2):
        raise NotImplementedError

    h = np.zeros([G,G])
    for i in range(1,G):
        h[0:i,i] = x[int((i-1)*i/2):int((i+1)*i/2)]
    Z = np.tanh(h)
    #Z = (np.exp(2*h)-1) / (np.exp(2*h) + 1)

    U = np.zeros([G,G])
    U[0,0] = 1
    U[0,1:G] = Z[0,1:G]
    for i in range(1,G):
        U[i,i] = U[i-1,i]*np.sqrt(1-Z[i-1,i]**2) / Z[i-1,i]
        U[i,(i+1):G] = Z[i,(i+1):G]*U[i-1,(i+1):G]*np.sqrt(1 - Z[i-1,(i+1):G]**2) / Z[i-1,(i+1):G] 
    return U

def nll(Psi,L,D,S,n):
    Cov = L @ (Psi @ L.T) + D
    R = np.linalg.solve(Cov,S)
    nll = n*np.log(np.linalg.det(2*np.pi*Cov))/2 + n*np.trace(R)/2
    return nll

def nll_efa(L,D,S,n):
    Cov = L @  L.T + D
    R = np.linalg.solve(Cov,S)
    nll = n*np.log(np.linalg.det(2*np.pi*Cov))/2 + n*np.trace(R)/2
    return nll

def Q_mat(J,G):
    Q = np.zeros([J,(1+G)])
    Q[:,0] = np.ones(J)
    for i in range(int(J/G)):
        Q[i*G:(i+1)*G, 1: (G+1)] = np.identity(G)
    return Q

def generator(J,G,Q,case):
    # case 1 pure bifactor model
    # case 2 approximate bifactor model
    if case == 'case1':
        D_true = np.identity(J)
        
        Bin = (1-2*rgt.binomial(1, 0.5,(J,G)))
        Sign = np.ones((J,1+G))
        Sign[:,1:] = Bin
        L1 = rgt.uniform(0.2,1,(J,(1+G)))
        L_true = np.where(Q ==1,L1*Sign,0)
        
        Phi_true = np.zeros([1+G,1+G])
        Phi_true[0,0] = 1
        #tmp = rgt.randn(int(G*(G-1)/2))
        
        tmp = rgt.uniform(-0.5,0.5,int(G*(G-1)/2))*(1-2*rgt.binomial(1,0.5,int(G*(G-1)/2)))
        
        Phi_true[1:(G+1),1:(G+1)] = stan_trans(tmp,G)
        Psi_true = Phi_true.T @ Phi_true
        Cov_true = L_true @ (Psi_true @ L_true.T) + D_true
    
    elif case == 'case2':
        D_true = np.identity(J)
        
        Bin = (1-2*rgt.binomial(1, 0.5,(J,G)))
        Sign = np.ones((J,1+G))
        Sign[:,1:] = Bin
        L1 = rgt.uniform(0.2,1,(J,(1+G)))
        L2 = rgt.uniform(0,0.1,(J,(1+G)))
        L_true = np.where(Q ==1,L1*Sign,L2*Sign)
        
        
        Phi_true = np.zeros([1+G,1+G])
        Phi_true[0,0] = 1
        #tmp = rgt.randn(int(G*(G-1)/2))
        tmp = rgt.uniform(-0.5,0.5,int(G*(G-1)/2))*(1-2*rgt.binomial(1,0.5,int(G*(G-1)/2)))
        
        Phi_true[1:(G+1),1:(G+1)] = stan_trans(tmp,G)
        Psi_true = Phi_true.T @ Phi_true
        Cov_true = L_true @ (Psi_true @ L_true.T) + D_true
        
    else:
        raise NotImplementedError('The case is not considered')
    
    return D_true,L_true,Phi_true,Psi_true,Cov_true

def sampling_process(J,Cov,n):
    mean = np.zeros(J)
    samples = np.random.multivariate_normal(mean, Cov, size=n)
    S = samples.T @ samples /n
    return S

def Bf_cons_pair(G):
    Pair = []
    for i in range(1,(G+1)):
        for j in range(i+1,(G+1)):
            Pair.append(np.array([i,j]))
    return Pair


def single_process(S,n,J,G,Pair,rho_0,rho_sigma,theta,gamma_0,tol,max_iter,Repeat,Min_point):
    #time_start = time.time()
    model = ebfa_solver(S)
    L_list = []
    NLL_list = []
    D_list = []
    Psi_list = []
    X_init_list = []
    UF_list = []
    for rep in range(Repeat):
        X_init_list.append(model.init_value(J,G))
    time_start = time.time()
    for rep in range(Repeat):
        x_est,iter_num,dist_val = model.alm_solve(X_init_list[rep],J,G,S,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter)
        if iter_num<max_iter:
            L_est,D_est,d_est,Psi_est,Cov_est = model.para_decompose(x_est,J,G)
            nll_est = nll(Psi_est,L_est,D_est,S,n)
            NLL_list.append(nll_est)
            L_list.append(L_est)
            D_list.append(D_est)
            Psi_list.append(Psi_est)
        else:
            UF_list.append(x_est)
    restart_num = 0
    while restart_num<5 and len(NLL_list)<Min_point:
        restart_num += 1
        U_len = len(UF_list)
        X_init_list = []
        for rep in range(U_len):
            X_init_list.append(model.init_value(J,G))
        UF_temp_list = []
        
        for rep in range(U_len):
            x_est,iter_num,dist_val = model.alm_solve(X_init_list[rep],J,G,S,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter)
                        
            if iter_num<max_iter:
                L_est,D_est,d_est,Psi_est,Cov_est = model.para_decompose(x_est,J,G)
                nll_est = nll(Psi_est,L_est,D_est,S,n)
                NLL_list.append(nll_est)
                L_list.append(L_est)
                D_list.append(D_est)
                Psi_list.append(Psi_est)
            else:
                UF_temp_list.append(x_est)
        UF_list = UF_temp_list 
        
    time_end = time.time()
    time_cost = time_end-time_start
    nll_list = np.array(NLL_list)
    nll_best = np.min(nll_list)
    loc_best = np.where(nll_list == nll_best)[0][0]
    L_alm = L_list[loc_best]
    Psi_alm = Psi_list[loc_best]
    D_alm = D_list[loc_best]
    finish_num = len(L_list)
    return L_alm,Psi_alm,D_alm,nll_best,finish_num,time_cost


def multiple_process(S,n,J,G,Pair,rho_0,rho_sigma,theta,gamma_0,tol,max_iter,Repeat,Round,Min_point):
    #time_start = time.time()
    model = ebfa_solver(S)
    L_list = []
    NLL_list = []
    D_list = []
    Psi_list = []
    UF_list = []
    time_start = time.time()
    X_init_list = []
    for rd in range(Round):
        for rep in range(Repeat):
            X_init_list.append(model.init_value(J,G))
    for rd in range(Round):
        params_list = [(X_init_list[i],J,G,S,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter) for i in range(rd*Repeat,(rd+1)*Repeat)]
        with Pool(processes= Repeat) as p:
            results = p.starmap(model.alm_solve, params_list)
            
        for x_est,iter_num,dist_val in results:
            if iter_num<max_iter:
                L_est,D_est,d_est,Psi_est,Cov_est = model.para_decompose(x_est,J,G)
                nll_est = nll(Psi_est,L_est,D_est,S,n)
                NLL_list.append(nll_est)
                L_list.append(L_est)
                D_list.append(D_est)
                Psi_list.append(Psi_est)
            else:
                UF_list.append(x_est)
    
    restart_num = 0
    while restart_num<5 and len(NLL_list)<Min_point:
        restart_num += 1
        U_len = len(UF_list)
        U_batch = 1 + U_len // Repeat
        UF_temp_list = []
        
        for rd in range(U_batch):
            tmp_num = min(Repeat,U_len-rd*Repeat)
            params_list = [(UF_list[i],J,G,S,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter) for i in range(rd*Repeat,rd*Repeat+tmp_num)]
            with Pool(processes= tmp_num) as p:
                results = p.starmap(model.alm_solve, params_list)
            
            for x_est,iter_num,dist_val in results:
                if iter_num<max_iter:
                    L_est,D_est,d_est,Psi_est,Cov_est = model.para_decompose(x_est,J,G)
                    nll_est = nll(Psi_est,L_est,D_est,S,n)
                    NLL_list.append(nll_est)
                    L_list.append(L_est)
                    D_list.append(D_est)
                    Psi_list.append(Psi_est)
                else:
                    UF_temp_list.append(x_est)
        UF_list = UF_temp_list
         
    
    
    time_end = time.time()
    time_cost = time_end-time_start
    nll_list = np.array(NLL_list)
    nll_best = np.min(nll_list)
    loc_best = np.where(nll_list == nll_best)[0][0]
    L_alm = L_list[loc_best]
    Psi_alm = Psi_list[loc_best]
    D_alm = D_list[loc_best]
    finish_num = len(L_list)
    return L_alm,Psi_alm,D_alm,nll_best,finish_num,time_cost

def BIC_ALM_process(S,n,J,G_min,G_max,rho_0,rho_sigma,theta,tol,max_iter,Repeat,Round,Min_point,multi=True):
    BIC_list = []
    L_est_list = []
    Psi_est_list = []
    D_est_list = []
    for g in range(G_min,G_max+1):
        Pair = Bf_cons_pair(g)
        gamma_0 = np.zeros([int((g-1)*g/2),J])
        if multi:
            L_alm,Psi_alm,D_alm,nll_alm,finish_num,time_cost = multiple_process(S,n,J,g,Pair,rho_0,rho_sigma,theta,gamma_0,tol,max_iter,Repeat,Round,Min_point)
        else:
            L_alm,Psi_alm,D_alm,nll_alm,finish_num,time_cost = single_process(S,n,J,g,Pair,rho_0,rho_sigma,theta,gamma_0,tol,max_iter,Repeat*Round,Min_point)
        BIC_list.append(2*nll_alm + np.log(n)*g*(g-1)/2)
        Psi_est_list.append(Psi_alm)
        L_est_list.append(L_alm)
        D_est_list.append(D_alm)
    BIC_arr = np.array(BIC_list)
    best_bic = np.min(BIC_arr)
    best_loc = np.where(BIC_arr == best_bic)[0][0]
    g_est = best_loc + G_min
    L_est = L_est_list[best_loc]
    Psi_est = Psi_est_list[best_loc]
    D_est = D_est_list[best_loc]
    
    return g_est,L_est,Psi_est,D_est,BIC_arr

def single_process_efa(S,n,J,G,Repeat):
    C = G+1
    model = ebfa_solver(S)
    L_list = []
    NLL_list = []
    D_list = []
    
    X_init_list = []
    for rep in range(Repeat):
        X_init_list.append(model.efa_init(J,C))
    
    for rep in range(Repeat):
        L_est,D_est,d_est = model.efa_solve(J,G,S,x_init = X_init_list[rep])
        nll_est = nll_efa(L_est,D_est,S,n)
        L_list.append(L_est)
        D_list.append(D_est)
        NLL_list.append(nll_est)
    NLL_arr = np.array(NLL_list)
    
    nll_best = np.min(NLL_arr)
    loc_best = np.where(NLL_arr == nll_best)[0][0]
    L_efa = L_list[loc_best]
    D_efa = D_list[loc_best]
    return L_efa,D_efa,nll_best
    
def multi_process_efa(S,n,J,G,Repeat,Round):
    C=G+1
    model = ebfa_solver(S)
    L_list = []
    NLL_list = []
    D_list = []
    
    X_init_list = []
    for rd in range(Round):
        for rep in range(Repeat):
            X_init_list.append(model.efa_init(J,C))
    
    for rd in range(Round):
        params_list = [(J,G,S,X_init_list[i]) for i in range(rd*Repeat,(rd+1)*Repeat)]
        with Pool(processes= Repeat) as p:
            results = p.starmap(model.efa_solve, params_list)
        
        for L_est,D_est,d_est in results:
            nll_est = nll_efa(L_est,D_est,S,n)
            L_list.append(L_est)
            D_list.append(D_est)
            NLL_list.append(nll_est)
    NLL_arr = np.array(NLL_list)
    nll_best = np.min(NLL_arr)
    loc_best = np.where(NLL_arr == nll_best)[0][0]
    L_efa = L_list[loc_best]
    D_efa = D_list[loc_best] 
    return L_efa,D_efa,nll_best

def BIC_EFA_process(S,n,J,G_min,G_max,Repeat,Round,multi=True):
    BIC_list = []
    L_est_list = []
    D_est_list = []
    
    for g in range(G_min,G_max+1):
        c = g+1 
        if multi:
            L_efa,D_efa,nll_efa = multi_process_efa(S,n,J,g,Repeat,Round)
        else: 
            L_efa,D_efa,nll_efa =single_process_efa(S,n,J,g,Repeat*Round)
        BIC_list.append(2*nll_efa + np.log(n)*(J*c-c*(c-1)/2))
        L_est_list.append(L_efa)
        D_est_list.append(D_efa)
    
    BIC_arr = np.array(BIC_list)
    best_bic = np.min(BIC_arr)
    best_loc = np.where(BIC_arr == best_bic)[0][0]
    g_est = best_loc + G_min
    L_est = L_est_list[best_loc]
    D_est = D_est_list[best_loc]
    
    return g_est,L_est,D_est,BIC_arr

def BIC_compare_process(S,n,J,G_min,G_max,rho_0,rho_sigma,theta,tol,max_iter,Repeat,Round,Min_point,multi=True):
    print('Start')
    g_alm,L_alm,Psi_alm,D_alm,BIC_alm = BIC_ALM_process(S,n,J,G_min,G_max,rho_0,rho_sigma,theta,tol,max_iter,Repeat,Round,Min_point,multi)
    g_efa,L_efa,D_efa,BIC_efa = BIC_EFA_process(S,n,J,G_min,G_max,Repeat,Round,multi)
    print('Finished')
    return g_alm,g_efa

def Bf_eval(L,Psi,D,Q,L_true,Psi_true,D_true,Q_true,P_list):
    G = L.shape[1]-1
    J = L.shape[0]
    # evaluate estimation error
    D_err = np.linalg.norm(D-D_true)**2/J
    
    L_err_list = []
    Psi_err_list = []
    for s in range(len(P_list)):
        Per_m = np.zeros([(1+G),(1+G)])
        Per_m[0,0] = 1
        Per_m[1:(1+G),1:(1+G)] = P_list[s] 
        L_rot = L_true @ Per_m
        S_rot = np.diag(np.diag(np.sign(L.T @ L_rot)))
        L_err_list[s] = np.linalg.norm(L - L_rot @ S_rot,ord='fro')**2/(J*(1+G))
            
        Psi_true_rot  = S_rot @ (Per_m.T @ (Psi_true @ (Per_m @ S_rot)))
        Psi_err_list[s] = np.linalg.norm(Psi -Psi_true_rot ,ord='fro')/(1+G)**2
    L_err = np.min(np.array(L_err_list))
    Psi_err = np.min(np.array(Psi_err_list))
    # evaluate structural error
    Emc = Exact_Cr(Q_true,Q,P_list)
    Acc = Average_Cr(Q_true,Q,P_list)
    return L_err,Psi_err,D_err,Emc,Acc

def single_process_vs(S,n,J,G,Pair,rho_0,rho_sigma,theta,gamma_0,tol,max_iter,Repeat,Min_point):
    #time_start = time.time()
    model = ebfa_solver(S)
    L_list = []
    NLL_list = []
    D_list = []
    Psi_list = []
    X_init_list = []
    UF_list = []
    time_start = time.time()
    for rep in range(Repeat):
        X_init_list.append(model.init_value(J,G))
    for rep in range(Repeat):
        x_est,iter_num,dist_val = model.alm_solve(X_init_list[rep],J,G,S,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter)
        if iter_num<max_iter:
            L_est,D_est,d_est,Psi_est,Cov_est = model.para_decompose(x_est,J,G)
            nll_est = nll(Psi_est,L_est,D_est,S,n)
            NLL_list.append(nll_est)
            L_list.append(L_est)
            D_list.append(D_est)
            Psi_list.append(Psi_est)
        else:
            UF_list.append(x_est)
    restart_num = 0
    Compute_num = Repeat
    while restart_num<5 and len(NLL_list)<Min_point:
        restart_num += 1
        U_len = len(UF_list)
        Compute_num = Compute_num + U_len
        X_init_list = []
        for rep in range(U_len):
            X_init_list.append(model.init_value(J,G))
        UF_temp_list = []
        
        for rep in range(U_len):
            x_est,iter_num,dist_val = model.alm_solve(X_init_list[rep],J,G,S,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter)
                        
            if iter_num<max_iter:
                L_est,D_est,d_est,Psi_est,Cov_est = model.para_decompose(x_est,J,G)
                nll_est = nll(Psi_est,L_est,D_est,S,n)
                NLL_list.append(nll_est)
                L_list.append(L_est)
                D_list.append(D_est)
                Psi_list.append(Psi_est)
            else:
                UF_temp_list.append(x_est)
        UF_list = UF_temp_list 
        
    time_end = time.time()
    time_cost = time_end-time_start
    nll_list = np.array(NLL_list)
    nll_best = np.min(nll_list)
    loc_best = np.where(nll_list == nll_best)[0][0]
    L_alm = L_list[loc_best]
    Psi_alm = Psi_list[loc_best]
    D_alm = D_list[loc_best]
    finish_num = len(L_list)
    return L_alm,Psi_alm,D_alm,nll_best,finish_num,time_cost,Compute_num


def multiple_process_vs(S,n,J,G,Pair,rho_0,rho_sigma,theta,gamma_0,tol,max_iter,Repeat,Round,Min_point):
    #time_start = time.time()
    model = ebfa_solver(S)
    L_list = []
    NLL_list = []
    D_list = []
    Psi_list = []
    UF_list = []
    time_start = time.time()
    X_init_list = []
    for rd in range(Round):
        for rep in range(Repeat):
            X_init_list.append(model.init_value(J,G))
    for rd in range(Round):
        params_list = [(X_init_list[i],J,G,S,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter) for i in range(rd*Repeat,(rd+1)*Repeat)]
        with Pool(processes= Repeat) as p:
            results = p.starmap(model.alm_solve, params_list)
            
        for x_est,iter_num,dist_val in results:
            if iter_num<max_iter:
                L_est,D_est,d_est,Psi_est,Cov_est = model.para_decompose(x_est,J,G)
                nll_est = nll(Psi_est,L_est,D_est,S,n)
                NLL_list.append(nll_est)
                L_list.append(L_est)
                D_list.append(D_est)
                Psi_list.append(Psi_est)
            else:
                UF_list.append(x_est)
    Compute_num = Round*Repeat
    restart_num = 0
    while restart_num<5 and len(NLL_list)<Min_point:
        restart_num += 1
        U_len = len(UF_list)
        Compute_num = Compute_num + U_len
        U_batch = 1 + U_len // Repeat
        UF_temp_list = []
        
        for rd in range(U_batch):
            tmp_num = min(Repeat,U_len-rd*Repeat)
            params_list = [(UF_list[i],J,G,S,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter) for i in range(rd*Repeat,rd*Repeat+tmp_num)]
            with Pool(processes= tmp_num) as p:
                results = p.starmap(model.alm_solve, params_list)
            
            for x_est,iter_num,dist_val in results:
                if iter_num<max_iter:
                    L_est,D_est,d_est,Psi_est,Cov_est = model.para_decompose(x_est,J,G)
                    nll_est = nll(Psi_est,L_est,D_est,S,n)
                    NLL_list.append(nll_est)
                    L_list.append(L_est)
                    D_list.append(D_est)
                    Psi_list.append(Psi_est)
                else:
                    UF_temp_list.append(x_est)
        UF_list = UF_temp_list
         
    
    
    time_end = time.time()
    time_cost = time_end-time_start
    nll_list = np.array(NLL_list)
    nll_best = np.min(nll_list)
    loc_best = np.where(nll_list == nll_best)[0][0]
    L_alm = L_list[loc_best]
    Psi_alm = Psi_list[loc_best]
    D_alm = D_list[loc_best]
    finish_num = len(L_list)
    return L_alm,Psi_alm,D_alm,nll_best,finish_num,time_cost,Compute_num
    

def Sen_process(S,n,J,G,Pair,rho_0,gamma_0,tol,max_iter,Repeat,Round,Min_point,Sigma_set,Theta_set,L_true,Psi_true,D_true,Q_true,P_list,multi=False):
    N1 = len(Sigma_set)
    N2 = len(Theta_set)
    
    L_err_list = np.zeros([N1,N2])
    Psi_err_list = np.zeros([N1,N2])
    D_err_list = np.zeros([N1,N2])
    EMC_list = np.zeros([N1,N2])
    ACC_list = np.zeros([N1,N2])
    Time_list = np.zeros([N1,N2])
    Compute_num_list = np.zeros([N1,N2])
    
    for i in range(N1):
        for j in range(N2):
            rho_sigma = Sigma_set[i]
            theta = Theta_set[j]
            
            if multi:
                L_alm,Psi_alm,D_alm,nll_ALM,finish_num,time_cost,Compute_num = multiple_process_vs(S,n,J,G,Pair,rho_0,rho_sigma,theta,gamma_0,tol,max_iter,Repeat,Round,Min_point) 
            else:
                L_alm,Psi_alm,D_alm,nll_ALM,finish_num,time_cost,Compute_num = single_process_vs(S,n,J,G,Pair,rho_0,rho_sigma,theta,gamma_0,tol,max_iter,Repeat*Round,Min_point) 
            
            Q_alm = np.where(np.abs(L_alm)>tol,1,0)
            Q_alm[:,0] = np.ones(J)
            
            
            L_err,Psi_err,D_err,Emc,Acc = Bf_eval(L_alm,Psi_alm,D_alm,Q_alm,L_true,Psi_true,D_true,Q_true,P_list)
            
            L_err_list[i,j] = L_err
            Psi_err_list[i,j] = Psi_err
            D_err_list[i,j] = D_err
            EMC_list[i,j] = Emc
            ACC_list[i,j] = Acc
            Time_list[i,j] = time_cost
            Compute_num_list[i,j] = Compute_num
    return L_err_list,Psi_err_list,D_err_list,EMC_list,ACC_list,Time_list,Compute_num_list
            

if __name__ == '__main__':
    rgt.seed(2024)
    
    J=15           # (J,G) = (15,3), (30,5) 
    G=3 
    case = 'case1' # case1,case2
    n=2000         # 500,2000  
    
    tol = 1e-2
    max_iter = 100 
    rho_0 = 1
    rho_sigma = 10
    theta = 0.25
    gamma_0 = np.zeros([int((G-1)*G/2),J])
    Pair = Bf_cons_pair(G)
    P_list = BF_permutation(G)
    Repeat=50
    Min_point = int(Repeat/2)
    multi = False
    
    theta = 0.25
    rho_sigma = 10
    
    #Epoch = 100
    Round =2
    Epoch = 50
    
    Q_true = Q_mat(J,G)
    D_true,L_true,Phi_true,Psi_true,Cov_true = generator(J,G,Q_true,case)
    
    
    Q_true_name = 'BF_minor_sim/Rot/Q_true_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) 
    L_true_name = 'BF_minor_sim/Rot/L_true_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) 
    D_true_name = 'BF_minor_sim/Rot/D_true_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) 
    Psi_true_name = 'BF_minor_sim/Rot/Psi_true_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) 
    
    np.save(Q_true_name,Q_true)
    np.save(L_true_name,L_true)
    np.save(D_true_name,D_true)
    np.save(Psi_true_name,Psi_true)
    
    S_list = []
    #for rep in range(Epoch):
    for rep in range(Epoch*Round):
        S_list.append(sampling_process(J,Cov_true,n))
    
    # params_list = [(S_list[i],n,J,G,Pair,rho_0,rho_sigma,theta,gamma_0,tol,max_iter,Repeat,Min_point) for i in range(Epoch)]
    # with Pool(processes= Epoch) as p:
    #     results = p.starmap(single_process, params_list)
    
    # iter_num = 0
    # for L_alm,Psi_alm,D_alm,nll_alm,finish_num,time_cost in results:
        
        
    #     L_name = 'BF_minor_sim/Rot/L_alm_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) + '_' + str(iter_num)
    #     np.save(L_name,L_alm)
        
    #     Psi_name = 'BF_minor_sim/Rot/Psi_alm_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) + '_' + str(iter_num)
    #     np.save(Psi_name,Psi_alm)
        
    #     D_name = 'BF_minor_sim/Rot/D_alm_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) + '_' + str(iter_num)
    #     np.save(D_name,D_alm)
        
        
    #     iter_num += 1
        
    # iter_num_efa = 0
    # params_efa_list = [(S_list[i],n,J,G,Repeat) for i in range(Epoch)]
    # with Pool(processes= Epoch) as p:
    #     results_efa = p.starmap(single_process_efa, params_efa_list)
        
    # for L_efa,D_efa,nll_efa_res in results_efa:
    #     L_efa_name = 'BF_minor_sim/Rot/L_efa_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) + '_' + str(iter_num_efa)
    #     np.save(L_efa_name,L_efa)
        
    #     D_efa_name = 'BF_minor_sim/Rot/D_efa_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) + '_' + str(iter_num_efa)
    #     np.save(D_efa_name,D_efa)
        
    #     L_csv_name = 'BF_minor_sim/Rot/L_efa_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) + '_' + str(iter_num_efa) + '.csv'
    #     L_pd = pd.DataFrame(L_efa)
    #     L_pd.to_csv(L_csv_name)
        
    #     D_csv_name = 'BF_minor_sim/Rot/D_efa_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) + '_' + str(iter_num_efa) + '.csv'
    #     D_pd = pd.DataFrame(D_efa)
    #     D_pd.to_csv(D_csv_name)
        
    #     iter_num_efa += 1
    iter_num = 0
    iter_num_efa = 0
    for r in range(Round):
        params_list = [(S_list[i],n,J,G,Pair,rho_0,rho_sigma,theta,gamma_0,tol,max_iter,Repeat,Min_point) for i in range(r*Epoch,(r+1)*Epoch)]
        with Pool(processes= Epoch) as p:
            results = p.starmap(single_process, params_list)
        
        for L_alm,Psi_alm,D_alm,nll_alm,finish_num,time_cost in results:
            
            
            L_name = 'BF_minor_sim/Rot/L_alm_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) + '_' + str(iter_num)
            np.save(L_name,L_alm)
            
            Psi_name = 'BF_minor_sim/Rot/Psi_alm_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) + '_' + str(iter_num)
            np.save(Psi_name,Psi_alm)
            
            D_name = 'BF_minor_sim/Rot/D_alm_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) + '_' + str(iter_num)
            np.save(D_name,D_alm)
            
            
            iter_num += 1
            
        params_efa_list = [(S_list[i],n,J,G,Repeat) for i in range(r*Epoch,(r+1)*Epoch)]
        with Pool(processes= Epoch) as p:
            results_efa = p.starmap(single_process_efa, params_efa_list)
            
        for L_efa,D_efa,nll_efa_res in results_efa:
            L_efa_name = 'BF_minor_sim/Rot/L_efa_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) + '_' + str(iter_num_efa)
            np.save(L_efa_name,L_efa)
            
            D_efa_name = 'BF_minor_sim/Rot/D_efa_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) + '_' + str(iter_num_efa)
            np.save(D_efa_name,D_efa)
            
            L_csv_name = 'BF_minor_sim/Rot/L_efa_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) + '_' + str(iter_num_efa) + '.csv'
            L_pd = pd.DataFrame(L_efa)
            L_pd.to_csv(L_csv_name)
            
            D_csv_name = 'BF_minor_sim/Rot/D_efa_' + str(J) + '_' +  str(G) + '_' + str(n) + '_' + str(case) + '_' + str(iter_num_efa) + '.csv'
            D_pd = pd.DataFrame(D_efa)
            D_pd.to_csv(D_csv_name)
            
            iter_num_efa += 1
        
    
        
    