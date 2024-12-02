# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 09:44:48 2023

@author: XYZW
"""
import numpy as np
import nelson_siegel_fitting as nsf
import scipy.optimize as opt
#%%
expiries = [1/12,2/12,3/12,6/12,9/12,1,2,3,5,7,10,20,30]
rates = [0.012,0.0135,0.015,0.017,0.019,0.023,0.025,0.029,0.034,0.039,
         0.041,0.046,0.053]

params_NS = nsf.nelson_siegel(expiries,rates)
params_NSS = nsf.nelson_siegel_svensson(expiries,rates)
params_df = nsf.nelson_siegel_svensson(expiries,rates,return_type = 'dataframe')

"""
The results of the Nelson Siegel curve are:
    5.17% = theta1
    -3.83% = theta2
    -1.95% = theta3
    1.49 = theta4 
    
    That means the short rate term is 5.17% = theta1+theta2, long term rate = theta2 
"""
#%%
rates_up = np.array(rates)+0.003
rates_down = np.array(rates)-0.003

params_NS_up = nsf.nelson_siegel(expiries,rates_up)
params_NS_down = nsf.nelson_siegel(expiries,rates_down)
params_NSS_up = nsf.nelson_siegel_svensson(expiries,rates_up)
params_NSS_down = nsf.nelson_siegel_svensson(expiries,rates_down)

rel_error_NSS_up = params_NSS_up/params_NSS-1
rel_error_NSS_down = params_NSS_down/params_NSS-1
rel_error_NS_up = params_NS_up/params_NS-1
rel_error_NS_down = params_NS_down/params_NS-1
#%%
sol_up1 = nsf.nelson_siegel_fitter(expiries,rates_up,method = 1)
sol_up2 = nsf.nelson_siegel_fitter(expiries,rates_up,method = 2)
sol_up3 = nsf.nelson_siegel_fitter(expiries,rates_up,method = 3)
sol_down1 = nsf.nelson_siegel_fitter(expiries,rates_down,method = 1)
sol_down2 = nsf.nelson_siegel_fitter(expiries,rates_down,method = 2)
sol_down3 = nsf.nelson_siegel_fitter(expiries,rates_down,method = 3)
#%%
"""
Nelson Siegel fitters.
Polynomial fitters
"""
nelson_siegel_fitter1 = lambda x: sum(np.array([nsf.nelson_siegel_function1(
    x[0],x[1],x[2],x[3],expiries[i])-rates[i] for i in range(len(rates))])**2)

nelson_siegel_fitter2 = lambda x:sum(np.array([nsf.nelson_siegel_function2(
    x[0],x[1],x[2],x[3],expiries[i])-rates[i] for i in range(len(rates))])**2)

nelson_siegel_fitter3 = lambda x:sum(np.array([nsf.nelson_siegel_function3(
    x[0],x[1],x[2],x[3],x[4],expiries[i])-rates[i] for i in range(len(rates))])**2)

f = lambda a,b,c,x:a+b*x+c*x**2
g = lambda a,b,c,d,x:a+b*x+c*x**2+d*x**3

def poly_fitter(expiries,rates,method = 1,tol = 1e-4):
    if method == 1:
        poly_fitter1 = lambda x:sum(np.array([f(x[0],x[1],x[2],expiries[i])-rates[i] 
                                    for i in range(len(rates))])**2)
        sol = opt.minimize(poly_fitter1,[1.,1.,1.],method = 'BFGS',tol = tol)
    elif method ==2:
        poly_fitter2 = lambda x:sum(np.array([f(x[0],x[1],x[2],expiries[i])-rates[i] 
                                    for i in range(len(rates))])**2)
        sol = opt.minimize(poly_fitter2,[1.,1.,1.,1.],method = 'BFGS',tol = tol)
    return sol

poly_fitter1 = lambda x:sum(np.array([f(x[0],x[1],x[2],expiries[i])-rates[i] 
                            for i in range(len(rates))])**2)

poly_fitter2 = lambda x:sum(np.array([g(x[0],x[1],x[2],x[3],expiries[i])-rates[i] 
                            for i in range(len(rates))])**2)

sol_poly1 = opt.minimize(poly_fitter1,[1.,1.,1.],method = 'BFGS')
sol_poly2 = opt.minimize(poly_fitter2,[1.,1.,1.,1.],method = 'BFGS')
sol_poly2_tol2 =  opt.minimize(poly_fitter2,[1.,1.,1.,1.],method = 'BFGS',tol = 1e-3)
sol_poly1_tol2 = opt.minimize(poly_fitter1,[1.,1.,1.],method = 'BFGS',tol = 1e-3)

sol_poly_up1 = poly_fitter(expiries,rates_up,method = 1,tol = 1e-3)
sol_poly_up2 = poly_fitter(expiries,rates_up,method = 2,tol = 1e-3)
rel_error_sol_up1 = sol_poly_up1['x']/sol_poly1_tol2['x']-1
rel_error_sol_up2 = sol_poly_up2['x']/sol_poly2_tol2['x']-1
#%%
sol1 = opt.minimize(nelson_siegel_fitter1,np.array([1.,1.,1.,1.]),
                    method = 'BFGS')['x']
sol2 = opt.minimize(nelson_siegel_fitter2,np.array([1.,1.,1.,1.]),
                    method = 'BFGS')['x']
sol3 = opt.minimize(nelson_siegel_fitter3,np.array([1.,1.,1.,1.,1.]),
                    method = 'BFGS')['x']
sol4 = nelson_siegel_fitter(expiries,rates,method = 4)

fitted_rates_NS1 = [nelson_siegel_function1(*sol1,x) for x in expiries]
fitted_rates_NS2 = [nelson_siegel_function2(*sol2,x) for x in expiries]
fitted_rates_NS3 = [nelson_siegel_function3(*sol3,x) for x in expiries]
fitted_rates_NS4 = [nelson_siegel_function4(*sol4,x) for x in expiries]
df1 = np.array([rates,fitted_rates_NS1,fitted_rates_NS2,
                fitted_rates_NS3,fitted_rates_NS4],ndmin = 2).T
print(np.linalg.norm(np.array(fitted_rates_NS1)-np.array(rates),2))
print(np.linalg.norm(np.array(fitted_rates_NS2)-np.array(rates),2))
print(np.linalg.norm(np.array(fitted_rates_NS3)-np.array(rates),2))
#%%

