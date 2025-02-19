# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:31:35 2024

@author: 888
"""


import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as rgt
from scipy.optimize import minimize
from Permutation import Hierarchy_permutation,Compare_Q,Average_Cr,Exact_Cr
from tqdm import tqdm
from multiprocessing import Pool

def cons_fun_vh(x,J,G,Pair_cons):
    p = len(Pair_cons)

    cons_val = np.zeros([p,J])
    for i in range(p):
        Z = Pair_cons[i]
        cons_val[i,:] = x[ Z[0]*J:  (Z[0]+1)*J] * x[Z[1]*J:  (Z[1]+1)*J]
    return cons_val

def con_gd_vh(L,J,G,Pairs,gamma,rho):
    L2 = L**2
    grad_L = np.zeros([J,(G+1)])
    for i in range(len(Pairs)):
        Z = Pairs[i]
        a = Z[0]
        b = Z[1]
        gm = gamma[i,:]
        grad_L[:,a] = grad_L[:,a] + gm*L[:,b] + rho*L[:,a]*L2[:,b]
        grad_L[:,b] = grad_L[:,b] + gm*L[:,a] + rho*L[:,b]*L2[:,a]
    grad = np.zeros(J*(2+G))
    for i in range(1+G):
        grad[i*J: (i+1)*J] = grad_L[:,i]
    return grad

def objective_function_vh(x,*args):
    J,G,S,gamma,rho,Pairs = args
    if len(x) != J*(2+G) :
        raise NotImplementedError
    
    x_rp = x[:(G+1)*J].reshape((G+1),J)
    L = x_rp.T
    
    D = np.diag(x[(G+1)*J: (G+2)*J]**2)

    Cov = L @ L.T + D
    R = np.linalg.solve(Cov,S)

    loss_part = np.log(np.linalg.det(2*np.pi*Cov))/2 + np.trace(R)/2

    cons_val = cons_fun_vh(x,J,G,Pairs)
    pen_part = np.sum(gamma*cons_val) + 0.5*rho*np.sum(cons_val**2)

    loss = loss_part + pen_part
    return loss

def alm_gd_vh(x,*args):
    J,G,S,gamma,rho,Pairs = args
    if len(x) != J*(2+G):
        raise NotImplementedError

    #L = np.zeros([J,(G+1)])
    #for i in range(G+1):
    #    L[:,i] = x[int(i*J): int((i+1)*J)]
    
    x_rp = x[:(G+1)*J].reshape((G+1),J)
    L = x_rp.T
    
    D = np.diag(x[(G+1)*J: (G+2)*J]**2)    
    d = x[(G+1)*J: (G+2)*J]
    Cov = L @ L.T + D

    inv_Cov = np.linalg.solve(Cov,np.identity(J))

    Sdw = inv_Cov @ (S @ (inv_Cov))
    diff = inv_Cov -Sdw

    cons_grad = con_gd_vh(L,J,G,Pairs,gamma,rho)
    
    nll_grad = np.zeros_like(x)
    
    dL = diff @ L
    for i in range(G+1):
        nll_grad[i*J: (i+1)*J] = dL[:,i]
    
    dd = np.diag(diff) * d
    nll_grad[(G+1)*J: (G+2)*J] = dd
    
    full_grad = nll_grad + cons_grad
    return full_grad

def para_decompose_vh(x,J,G):
    x_rp = x[:(G+1)*J].reshape((G+1),J)
    L = x_rp.T
    d = x[(G+1)*J:(G+2)*J]
    D = np.diag(d)
    Cov = L @ L.T + D
    return L,d,Cov

def Hierarchy_solve(x_init,J,G,S,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter):
    x_old = x_init.copy()
    
    gamma = gamma_0.copy()
    rho = rho_0
       
    L_old,d_old,Cov_old = para_decompose_vh(x_old,J,G)
    cons_val_old = cons_fun_vh(x_old,J,G,Pair)
    
    dist_val = np.max(np.sort(np.abs(L_old[:,1:]),axis=1)[:,-3])
    #print(dist_val)
    dist_norm = np.linalg.norm(x_old)/np.sqrt(4*J)
    iter_num = 0
    while max(dist_val,dist_norm) > tol and iter_num < max_iter:
        result = minimize(objective_function_vh,x_old,args=(J,G,S,gamma,rho,Pair),method = 'L-BFGS-B',jac = alm_gd_vh)
        x_new = result.x
            
        cons_val_new =  cons_fun_vh(x_new,J,G,Pair)
        gamma = gamma + rho *cons_val_new
        if np.linalg.norm(cons_val_old,ord = 'fro') > theta*np.linalg.norm(cons_val_new,ord = 'fro'):
            rho = rho*rho_sigma
        dist_norm = np.linalg.norm(x_old-x_new)/np.sqrt(4*J) 
        x_old = x_new.copy()
        L_old,d_old,Cov_old = para_decompose_vh(x_old,J,G)
        dist_val = np.max(np.sort(np.abs(L_old[:,1:]),axis=1)[:,-3])
        #print(dist_val)
        iter_num = iter_num + 1
    return x_old,iter_num,dist_val

def init_value_vh(J,G):
    x = np.zeros((G+2)*J)
    
    x[J*(G+1): J*(G+2)] = 1 + rgt.rand(J)
    
    x[: J] = 1 + rgt.rand(J)
    
    x[J : J*(G+1)] = rgt.randn(G*J)
    
    return x

def Hier_Q(J):
    if J % 4:
        raise NotImplementedError
    Q = np.zeros([J,7])
    Q[:,0] = np.ones(J)
    Q[:int(J/2),1] = np.ones(int(J/2))
    Q[int(J/2):J,2] = np.ones(int(J/2))
    Q[:int(J/4),3] = np.ones(int(J/4))
    Q[int(J/4):int(J/2),4] = np.ones(int(J/4))
    Q[int(J/2):int(3*J/4),5] = np.ones(int(J/4))
    Q[int(3*J/4):J,6] = np.ones(int(J/4))
    return Q

def generator_vh(J,G,Q):
    #D_true = np.diag(1+2*rgt.rand(J))
    D_true = np.identity(J)
    L_true =np.zeros([J,(1+G)])
    
    Bin = (1-2*rgt.binomial(1, 0.5,(J,G)))
    Sign = np.ones((J,1+G))
    Sign[:,1:] = Bin
    
    X = rgt.uniform(0.2,1,(J,(1+G)))
    L_true = X*Sign*Q
    
    Cov_true = L_true @ L_true.T + D_true
    return D_true,L_true,Cov_true

def sampling_process(J,Cov,n):
    mean = np.zeros(J)
    samples = np.random.multivariate_normal(mean, Cov, size=n)
    S = samples.T @ samples /n
    return S

def constraint_pair_form(Z_prev,pre_num,append_num):
    Z = Z_prev.copy()
    
    col_full = np.arange(pre_num+append_num)
    col_prev = np.arange(pre_num)
    
    col_append = col_full[pre_num:]
    
    layer_append = np.floor(np.log2(col_append+1)) + 1
    
    if len(np.unique(layer_append)) != 1:
          raise NotImplementedError                  
    
    set_prev = set(col_prev[1:])
    for i in range(append_num):
        a = col_append[i]
        forward_list = []
        forward_loc = a
        layer = int(layer_append[i])
        for j in range(layer-2):
            forward_loc = int((forward_loc-1)/2)
            forward_list.append(forward_loc)
        set_forward = set(forward_list)
        set_con = set_prev - set_forward
        con_np = np.array(list(set_con))
        
        for j in range(len(con_np)):
            b = con_np[j]
            cons = np.array([a,b])
            Z = np.vstack((Z,cons))
        for j in range(i+1,append_num):
            b = col_append[j]
            cons = np.array([a,b])
            Z = np.vstack((Z,cons))
            
    return Z

def multi_start_opt(J,G,S,Z,gamma_0,rho_0,rho_sigma,theta,tol,max_iter,Repeat,it):
    print('Start')
    Nll_list = np.zeros(Repeat)
    Niter_list = np.zeros(Repeat)
    X_list = []
    for i in range(Repeat):
        x_init_alm = init_value_vh(J,G)
        x_alm,iter_num,dist_val =  Hierarchy_solve(x_init_alm,J,G,S,Z,gamma_0,rho_0,rho_sigma,theta,tol,max_iter)
        nll = objective_function_vh(x_alm,J,G,S,np.zeros([len(Z),J]),0,Z)
        Nll_list[i] = nll
        Niter_list[i] = iter_num
        X_list.append(x_alm)
        
    opt_list = np.where(Niter_list<max_iter)[0]
    if len(opt_list)>0:
        best_val = np.min(Nll_list[opt_list])
        best_loc = np.where(Nll_list == best_val)[0][0]
        x_est = X_list[best_loc]
        L_est,d_est,Cov_est = para_decompose_vh(x_est,J,G)
        UF_id = 0
    else:
        UF_id = 1
        
        best_val = np.min(Nll_list)
        best_loc = np.where(Nll_list == best_val)[0][0]
        x_est = X_list[best_loc]
        L_est,d_est,Cov_est = para_decompose_vh(x_est,J,G)
    print('Finished')
    
    return L_est,UF_id,it,d_est
    

if __name__ == '__main__':
    rgt.seed(2024)
    tol =1e-2
    rho_0 = 1
    rho_sigma = 10
    theta = 0.25
    max_iter = 100
    n = 500
    Epoch = 100
    Repeat = 20
    G=6
    J=40 
    
    Permutation_list = Hierarchy_permutation()
    
    Z = np.array([[1,2]])
    pre_num = 3
    append_num = 2
    for i in range(2):
        Z = constraint_pair_form(Z,pre_num,append_num)
        pre_num = pre_num + append_num
        
    gamma_0 = np.zeros([len(Z),J])
       
    Q = Hier_Q(J)
    
    Exact_cover = []
    Average_cover = []
    
    D_true,L_true,Cov_true = generator_vh(J,G,Q)
    
    L_name = 'BF_minor_sim/Hier/L_true_' + str(J) + '_' + str(n) 
    np.save(L_name,L_true)
    
    S_list = []
    for epoch in tqdm(range(Epoch)):
        S_list.append(sampling_process(J,Cov_true,n))
        
    params_list = [(J,G,S_list[i],Z,gamma_0,rho_0,rho_sigma,theta,tol,max_iter,Repeat,i) for i in range(Epoch)]
    
    with Pool(processes= Epoch) as p:
        results = p.starmap(multi_start_opt, params_list)
    
    UF_list = []
    Exact_cover_list = []
    Average_cover_list = []
    for L_est,UF_id,it,d_est in results:
        UF_list.append(UF_id)
        Q_raw = np.where(np.abs(L_est)>tol,1,0)
        Q_raw[:,0] = np.ones(J)
        
        Exact_cover_list.append(Exact_Cr(Q,Q_raw,Permutation_list))
        Average_cover_list.append(Average_Cr(Q,Q_raw,Permutation_list))
        
        L_est_name = 'BF_minor_sim/Hier/L_est_' + str(J) + '_' + str(n) + '_' + str(it)
        np.save(L_est_name,L_est)
        
        d_est_name = 'BF_minor_sim/Hier/d_est_' + str(J) + '_' + str(n) + '_' + str(it)
        np.save(d_est_name,d_est)
        
    print(UF_list)
    print(Exact_cover_list)
    print(Average_cover_list)
    
    UF_list = np.array(UF_list)
    Exact_cover_list = np.array(Exact_cover_list)
    Average_cover_list = np.array(Average_cover_list)
    
    print(np.mean(UF_list))
    print(np.mean(Exact_cover_list))
    print(np.mean(Average_cover_list))
    
    Exact_cover_name = 'BF_minor_sim/Hier/EMC_' + str(int(J)) + '_' + str(int(n))
    np.save(Exact_cover_name,Exact_cover_list)
    
    Average_cover_name = 'BF_minor_sim/Hier/ACC_' + str(int(J)) + '_' + str(int(n))
    np.save(Average_cover_name,Average_cover_list)
    
    UF_name = 'BF_minor_sim/Hier/UF_' + str(int(J)) + '_' + str(int(n))
    np.save(UF_name,UF_list)
        
        
    