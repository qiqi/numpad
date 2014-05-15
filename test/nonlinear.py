# Copyright Qiqi Wang (qiqi@mit.edu) 2013

import sys
from pylab import *
from numpy import *

sys.path.append('..')

from lssode import *
from numpad import *

def lorenz(u, rho):
    shp = u.shape
    x, y, z = u.reshape([-1, 3]).T
    sigma, beta = 10, 8./3
    dxdt, dydt, dzdt = sigma*(y-x), x*(rho-z)-y, x*y - beta*z
    return transpose([dxdt, dydt, dzdt]).reshape(shp)

def vanderpol(u, mu):
    shp = u.shape
    u = u.reshape([-1,2])
    dudt = zeros(u.shape, u.dtype)
    dudt[:,0] = u[:,1]
    dudt[:,1] = -u[:,0] + mu * (1 - u[:,0]**2) * u[:,1]
    return dudt.reshape(shp)

def costfunction(u,mu):
    return u[:,1]**8

CASE = 'vanderpol'

if CASE == 'vanderpol':
    mus = linspace(0.2, 2.0, 2)
#    mus = linspace(0.2,0.2,1)
    # x0 = random.rand(2)
    x0 = array([0.5, 0.5])
    dt, T = 0.01, 100
    t = 30 + dt * arange(int(T / dt))
    
    solver = lssSolver(vanderpol, x0, mus[0], t)
#    J=solver.evaluate(costfunction)
#    J=pow(J, 1./8)
#    print J
    u, t = [solver.u.copy()], [solver.t.copy()]
    
    for mu in mus[1:]:
        print('mu = ', mu)
        # solver.u[0,0] += 1E-6
        solver = lssSolver(vanderpol, solver.u, mus[0], solver.t)
        solver.lss(mu)
        u.append(solver.u.copy())
        t.append(solver.t.copy())
    
    u, t = array(u), array(t)
    
#    figure(figsize=(5,10))
#    contourf(mus[:,newaxis] + t * 0, t, u[:,:,0], 501)
#    ylim([t.min(1).max(), t.max(1).min()])
#    xlabel(r'$\mu$')
#    ylabel(r'$t$')
#    title(r'$x$')
#    colorbar()
#    show()

elif CASE == 'lorenz':
    rhos = linspace(28, 33, 21)
    x0 = random.rand(3)
    dt, T = 0.01, 30
    t = 30 + dt * arange(int(T / dt))
    
    solver = lssSolver(lorenz, x0, rhos[0], t)
    u, t = [solver.u.copy()], [solver.t.copy()]
    
    for rho in rhos[1:]:
        print('rho = ', rho)
        solver.lss(rho)
        u.append(solver.u.copy())
        t.append(solver.t.copy())
    
    u, t = array(u), array(t)
    
    figure(figsize=(5,10))
    contourf(rhos[:,newaxis] + t * 0, t, u[:,:,2], 501)
    ylim([t.min(1).max(), t.max(1).min()])
    xlabel(r'$\rho$')
    ylabel(r'$t$')
    title(r'$z$')
    colorbar()
    show()


