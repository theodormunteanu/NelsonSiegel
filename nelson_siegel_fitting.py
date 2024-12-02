# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:06:05 2023

@author: XYZW
"""

import numpy as np
import scipy.optimize as opt
import pandas as pd
#%%
def nelson_siegel_function1(theta1,theta2,theta3,theta4,T,t = 0):
    """
    In the Nelson Siegel model, there are 4 params to be estimated
    
    T-t is time to expiry
    """
    q1 = np.exp(-(T-t)/theta4)
    q2 = (1-q1)/((T-t)/theta4)
    return theta1+theta2*q2+theta3*(q2-q1)

def nelson_siegel_function2(b1,b2,b3,lbd,t):
    """
    Same Nelson Siegel model as above but with t as time to expiry
    """
    return b1+b2*(1-np.exp(-lbd*t))/(lbd*t)+b3*((1-np.exp(-lbd*t))/(lbd*t)-np.exp(-lbd*t))

def nelson_siegel_function3(b1,b2,b3,lbd1,lbd2,t):
    
    """
    Nelson Siegel with 2 decay parameters
    
    A more relaxed with decay parameter lbd replaced with lambda1 in the first term and
    lambda 2 in the second parameter.
    """
    return b1+b2*(1-np.exp(-lbd1*t)/(lbd1*t))+\
        b3*((1-np.exp(-lbd2*t))/(lbd2*t) - np.exp(-lbd2*t))

def nelson_siegel_function4(b1,b2,b3,lbd1,lbd2,lbd3,t):
    """
    Nelson Siegel with 3 decay parameters. 
    
    An even more relaxed model, with lbd1 for the first bracket, lambda2 for the first half of the second
    and lambda3 decay parameter.
    """
    return b1+b2*(1-np.exp(-lbd1*t)/(lbd1*t))+\
        b3*((1-np.exp(-lbd2*t))/(lbd2*t) - np.exp(-lbd3*t))
        
def NSS_fct(beta0,beta1,beta2,beta3,tau1,tau2,x):
    """
    beta0,beta1,beta2,beta3: level, slope, curvature and hump parameters
    
    tau1,tau2: decay parameters
    
    """
    return beta0+beta1*(1-np.exp(-x/tau1))/(x/tau1)+beta2*((1-np.exp(-x/tau1))/(x/tau1)-np.exp(-x/tau1))+\
        beta3*((1-np.exp(-x/tau2))/(x/tau2)-np.exp(-x/tau2))

def nelson_siegel_fitter(expiries,rates,method = 1,tol  =1e-4):
    """
    The function encapsulates all 3 Nelson Siegel models from above. 
    
    The first 2 nelson siegel functions are equivalent for t = 0. 
    
    The third NS function has 2 decaying parameters
    
    The fourth NS function has 3 decaying parameters. 
    
    The fifth NS function is Nelson Siegel Svensson model. 
    
    Returns:
        The parameters calibrating the nodes (expiries, rates)
    """
    if method==1:
        fun = lambda x: sum(np.array([nelson_siegel_function1(
            x[0],x[1],x[2],x[3],expiries[i])-rates[i] 
            for i in range(len(rates))])**2)
        sol = opt.minimize(fun,np.array([1.,1.,1.,1.]),method = 'BFGS')['x']
    elif method==2:
        fun = lambda x: sum(np.array([nelson_siegel_function2(
            x[0],x[1],x[2],x[3],expiries[i])-rates[i] 
            for i in range(len(rates))])**2)
        sol = opt.minimize(fun,np.array([1.,1.,1.,1.]),method = 'BFGS')['x']
    elif method==3:
        fun = lambda x:sum(np.array([nelson_siegel_function3(
            x[0],x[1],x[2],x[3],x[4],expiries[i])-rates[i] 
            for i in range(len(rates))])**2)
        sol = opt.minimize(fun,np.array([1.,1.,1.,1.,1.]),method = 'BFGS')['x']
    elif method==4:
        fun = lambda x:sum(np.array([nelson_siegel_function4(
            x[0],x[1],x[2],x[3],x[4],x[5],expiries[i])-rates[i] 
            for i in range(len(rates))])**2)
        sol = opt.minimize(fun,np.array([1.,1.,1.,1.,1.,1.]),method = 'BFGS')['x']
    elif method ==5:
        NSS_func = lambda beta0,beta1,beta2,beta3,tau1,tau2,x: beta0+beta1*(
            1-np.exp(-x/tau1))/(x/tau1)+beta2*((1-np.exp(-x/tau1))/(x/tau1)-np.exp(-x/tau1))+\
            beta3*((1-np.exp(-x/tau2))/(x/tau2)-np.exp(-x/tau2))
        fun = lambda x:sum(np.array([NSS_func(*x,expiries[i])-rates[i] 
                    for i in range(len(rates))])**2)
        sol = opt.minimize(fun,np.array([1]*6),method = 'BFGS',tol = tol)['x']
    return sol

def nelson_siegel_svensson(expiries,rates,tol = 1e-4,return_type = 'array'):
    """
    Given the Nelson Siegel Svensson model find the parameters.
    
    Actually, here we use 2 decay parameters but with four betas instead of 3. 
    
    For the beta1, beta2 we have one decay. 
    
    There are 6 parameters to be found.
    """
    NSS_func = lambda beta0,beta1,beta2,beta3,tau1,tau2,x: beta0+beta1*(
        1-np.exp(-x/tau1))/(x/tau1)+beta2*((1-np.exp(-x/tau1))/(x/tau1)-np.exp(-x/tau1))+\
        beta3*((1-np.exp(-x/tau2))/(x/tau2)-np.exp(-x/tau2))
    fun = lambda x:sum(np.array([NSS_func(*x,expiries[i])-rates[i] 
                for i in range(len(rates))])**2)
    if return_type == 'array':
        sol = opt.minimize(fun,np.array([1]*6),method = 'BFGS',tol = tol)['x']
        return sol
    else:
        return opt.minimize(fun,np.array([1]*6),method = 'BFGS',tol = tol)

def nelson_siegel(expiries,rates,tol  =1e-4,return_type = 'array'):
    """
    Returns: 
        theta1: level parameter
        
        theta2: rotation parameter
        
        theta3: shape
        
        theta4: location of the breaking point
    
    NS_func is the nelson siegel function where x is the maturity and actually it is Nelson Siegel model
    
    NS_func could be defined as nelson_siegel_function2 instead of manually defining as below
    """
    NS_func = lambda beta0,beta1,beta2,tau1,x: beta0+beta1*(
        1-np.exp(-x/tau1))/(x/tau1)+beta2*((1-np.exp(-x/tau1))/(x/tau1)-np.exp(-x/tau1))
    fun = lambda x:sum(np.array([NS_func(*x,expiries[i])-rates[i] for i in range(len(rates))])**2)
    if return_type == 'array':
        sol = opt.minimize(fun,np.array([1]*4),method = 'BFGS',tol = tol)['x']
        return sol
    else:
        return opt.minimize(fun,np.array([1]*4),method = 'BFGS',tol = tol)
