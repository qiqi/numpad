# Copyright Qiqi Wang (qiqi@mit.edu) 2013

import sys
from pylab import *
from numpy import *

set_printoptions(threshold=nan)

sys.path.append('..')

from lssode import *
from numpad import *


def outputVector1d(vec,size,filename):
    ufile=open(filename,'w')
    for i in range(size[0]):
      ufile.write('%.40f \n' %(vec[i]))
    ufile.close()
    print('File written: ' +filename)

def outputVector2d(vec,size,filename):
    ufile=open(filename,'w')
    for i in range(size[0]):
      ufile.write('%.40f %.40f \n' %(vec[i,0],vec[i,1]))
    ufile.close()
    print('File written: ' +filename)


import resource
def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           '''%(point,usage[0],usage[1],
                (usage[2]*resource.getpagesize())/1000000.0 )


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

#open files for output
blackboxfile=open('blackbox.dat','w')
redgradfile=open('piggyback.dat','w')
#G1sfile=open('G1_s.dat','w')
#G1sfile.close()
#G2sfile=open('G2_s.dat','w')
#G2sfile.close()
file1=open('adj_update.dat','w')
file1.close()



if CASE == 'vanderpol':
#    mus = linspace(0.2, 2.0, 10)
    mus = linspace(0.2,0.2,2)
    # x0 = random.rand(2)
    x0 = array([0.5, 0.5])
    dt, T = 0.01, 100
    tmp=int(T / dt)
    t = 30 + dt * arange(tmp)

    u_adj=array([0.0, 0.0])
    dt_adj=0.0
    
    solver = lssSolver(vanderpol, x0, mus[0], t, dt, u_adj,dt_adj)
    u, t = [solver.u.copy()], [solver.t.copy()]

    for mu in mus[1:]:
        print('mu = ', mu)
        
        #print('perturb mu')
        #mu=mu+1E-6

        
        for iNewton in range(8):
            #if iNewton == 5:
            #   print('perturb u')
            #   solver.u[0,0]+= 1E-6
            #   solver.dt[0]+= 1E-6
            #   print('perturb mu')
            #   mu=mu+1E-6
        
            solver = lssSolver(vanderpol, base(solver.u), mu, base(solver.t),base(solver.dt), \
                        base(solver.u_adj), base(solver.dt_adj))
            
            
            solver.lss(mu,maxIter=1,disp=True, counter=iNewton)

            print('reduced gradient %.40f ' %solver.redgrad)
            redgradfile.write('%.40f \n'%solver.redgrad)

            
            #print('Jdiffmu %.40f ' %solver.J.diff(mu))
            #blackboxfile.write('%.40f \n' %solver.J.diff(mu))

            solver.t[1:] = solver.t[0] + np.cumsum(base(solver.dt))
            
            print(using('newton'+str(iNewton)))
            #print('J '+str(solver.J))
        u.append(base(solver.u).copy())
        t.append(base(solver.t).copy())


    outputVector2d(solver.u,solver.u.shape,'u.dat')
    outputVector1d(solver.t,solver.t.shape, 't.dat')
    #ufile=open('u.dat','w')
    #ufile.write('mu = '+str(mu)+'\n')
    #ufile.write(str(base(solver.u)))
    #ufile.close()


    u, t = array(u), array(t)
   
#    figure(figsize=(5,10))
#    contourf(mus[:,newaxis] + t * 0, t, u[:,:,0], 501)
#    ylim([base(t).min(1).max(), base(t).max(1).min()])
#    xlabel(r'$\mu$')
#    ylabel(r'$t$')
#    title(r'$x$')
#    colorbar()
#    show(block=True)

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


#close output files
blackboxfile.close()
redgradfile.close()
