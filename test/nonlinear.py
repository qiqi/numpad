# Copyright Qiqi Wang (qiqi@mit.edu) 2013

import sys
from pylab import *
from numpy import *
from scipy.interpolate import interp1d


#set_printoptions(threshold=nan)

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

def outputVector3d(vec,size,filename):
    ufile=open(filename,'w')
    for i in range(size[0]):
      ufile.write('%.40f %.40f %.40f\n' %(vec[i,0],vec[i,1],vec[i,2]))
    ufile.close()
    print('File written: ' +filename)


import struct
def outputBinary(vec,size,filename):
    binfile=open(filename,'wb')
    for i in range(size):
      data=struct.pack('d',vec[i])
      binfile.write(data)
    binfile.close()
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

def costfunction(u,target,mu):
    J=(u[:,1]**8).mean(0)
    J=J**(1./8)
    J=1./2*(J-target)**2
    return J

CASE = 'vanderpol'

#open files for output
#blackboxfile=open('blackbox.dat','w')
redgradfile=open('redgrad.dat','w')
adjresfile=open('adj_res.dat','w')
#adjnextfile=open('adj_next.dat','w')
primresfile=open('prim_res.dat', 'w')




if CASE == 'vanderpol':
#    mus = linspace(0.2, 2.0, 10)
    mus = linspace(0.2,1.0,2)
    # x0 = random.rand(2)
    x0 = array([0.5, 0.5])
    dt, T = 0.01,0.05 
    tmp=int(T / dt)
    t = 30 + dt * arange(tmp)

    #tfix=t.copy()

    u_adj=array([0.0, 0.0])
    dt_adj=0.0
    

    solver = lssSolver(vanderpol, x0, mus[0], t, dt, u_adj, dt_adj)
    u, t = [solver.u.copy()], [solver.t.copy()]

    
    print(solver.dt.shape, solver.t.shape) 
    for mu in mus[1:]:
       
        for iNewton in range(6):

            ubase = array(base(solver.u))
            tbase = array(base(solver.t))
            dtbase = array(base(solver.dt))
            mubase = array(base(mu))
            u_adjbase = array(base(solver.u_adj))
            dt_adjbase = array(base(solver.dt_adj))
 
            #compute primal update
            solver = lssSolver(vanderpol, ubase, mubase, tbase, dtbase, u_adjbase, dt_adjbase)
            [G1,G2,prim_res] = solver.lss(mubase, maxIter=1,disp=True, counter=iNewton)

            #evaluate costfunction
            solver.J = costfunction(ubase,solver.target,mubase)
            print('J %.40f' %solver.J) 


           
 
            #compute Jacobi of (G1,G2) wrt (u,dt) componentwise in 5th iteration
            if iNewton == 5:
               ## G2 wrt dt:
               #for i in range(solver.dt.shape[0]):
               #  unit=zeros(solver.dt.shape)
               #  unit[i]=1.0
               #  G2unit_dt = array((G2 * unit).sum().diff(dtbase)).reshape(solver.dt_adj.shape)
               #  print(G2unit_dt)
               #  outputBinary(G2unit_dt,G2unit_dt.shape[0],'binaryN5/G2unit_dt'+str(i)+'.bin')
               #  #print(G2unit_dt)
               #print('test',array(G2.diff(dtbase)))
     
               ## G2 wrt u:
               #for i in range(solver.dt.shape[0]):
               #  unit=zeros(solver.dt.shape)
               #  unit[i]=1.0
               #  G2unit_u = array((G2 * unit).sum().diff(ubase)) #.reshape(solver.dt_adj.shape)
               #  G2unit_u=transpose(G2unit_u)
               #  print(G2unit_u)
               #  outputBinary(G2unit_u,G2unit_u.shape[0],'binaryN5/G2unit_u'+str(i)+'.bin')
               #print('test', array(G2.diff(ubase)))
     
     
               ## G1 wrt u:
               #for i in range(solver.u.shape[0]):
               #  unit=zeros(solver.u.shape)
               #  unit[i,0]=1.0
               #  G1unit_u = array((G1 * unit).sum().diff(ubase)) #.reshape(u.shape)
               #  G1unit_u=transpose(G1unit_u)
               #  print(G1unit_u)
               #  outputBinary(G1unit_u,G1unit_u.shape[0],'binaryN5/G1unit_u1_'+str(i)+'.bin')
     
               #  unit=zeros(solver.u.shape)
               #  unit[i,1]=1.0
               #  G1unit_u = array((G1 * unit).sum().diff(ubase)) #.reshape(u.shape)
               #  G1unit_u=transpose(G1unit_u)
               #  print(G1unit_u)
               #  outputBinary(G1unit_u,G1unit_u.shape[0],'binaryN5/G1unit_u2_'+str(i)+'.bin')
               #print('test', array(G1.diff(ubase)))
              
     
               # G1 wrt dt:
               for i in range(solver.u.shape[0]):
                 unit=zeros(solver.u.shape)
                 unit[i,0]=1.0
                 G1unit_dt = array((G1 * unit).sum().diff(dtbase)).reshape(solver.dt.shape)
                 #G1unit_u=transpose(G1unit_u)
                 print(G1unit_dt)
                 outputBinary(G1unit_dt,G1unit_dt.shape[0],'binaryN5/G1unit_dt1_'+str(i)+'.bin')
     
                 unit=zeros(solver.u.shape)
                 unit[i,1]=1.0
                 G1unit_dt = array((G1 * unit).sum().diff(dtbase)).reshape(solver.dt.shape)
                 #G1unit_dt=transpose(G1unit_dt)
                 print(G1unit_dt)
                 outputBinary(G1unit_dt,G1unit_dt.shape[0],'binaryN5/G1unit_dt2_'+str(i)+'.bin')
               print('test', array(G1.diff(dtbase)))


            
            ##compute adjoint update
            #J_u = array((solver.J).diff(ubase).todense()).reshape(solver.u_adj.shape)
            ##print('J_u...',J_u)
            #G1_u = array((G1 * solver.u_adj).sum().diff(ubase)).reshape(solver.u_adj.shape)
            ##print('G1_u...',G1_u)
            #G2_u = array((G2 * solver.dt_adj).sum().diff(ubase)).reshape(solver.u_adj.shape)
            ##print('G2_u...', G2_u)
            #G1_dt = array((G1 * solver.u_adj).sum().diff(dtbase)).reshape(solver.dt_adj.shape)
            ##print('G1_dt...', G1_dt)
            #G2_dt = array((G2 * solver.dt_adj).sum().diff(dtbase)).reshape(solver.dt_adj.shape)
            ##print('G2_dt...', G2_dt)
            #
            #u_adj_next =  J_u \
            #            + G1_u \
            #            + G2_u
            #dt_adj_next = G1_dt \
            #            + G2_dt
           
            #compute residuals
            #adj_res = (ravel(u_adj_next - solver.u_adj)**2).sum() + (ravel(dt_adj_next - solver.dt_adj)**2).sum()
            #adj_res =  (ravel(dt_adj_next - solver.dt_adj)**2).sum()
            #adj_res = sqrt(adj_res)
            prim_res = sqrt((ravel(prim_res)**2).sum()) 
            #if iNewton == 0:
            #    adj_res0 = adj_res
            #    prim_res0 = prim_res
            #adj_res = adj_res  / adj_res0
            #prim_res = prim_res  / prim_res0
            #adjresfile.write('%.40f \n' %(adj_res))
            primresfile.write('%.40f \n' %(prim_res))

            
            #print('adjoint residuum %.40f' %adj_res)
            
            

            #update primal
            solver.u = G1
            solver.dt = G2
            solver.t[1:] = solver.t[0] + cumsum(solver.dt)


            ##update adjoint
            #if iNewton > 5:
            #  solver.u_adj =  u_adj_next
            #  solver.dt_adj = dt_adj_next


            ##projection
            #if iNewton > -1:
            #     dudt = solver.f(solver.u,mu)
            #     pr = (dudt*solver.u_adj).sum(1) / (dudt*dudt).sum(1)
            #     solver.u_adj = solver.u_adj - (dudt*pr[:,newaxis])
                       
            #
            ##compute reduced gradient
            #G1_s = array((G1 * solver.u_adj).sum().diff(mubase))
            #G2_s = array((G2 * solver.dt_adj).sum().diff(mubase))
            #redgrad = G1_s + G2_s
            #print('reduced gradient %.40f ' %redgrad)
            #redgradfile.write('%.40f \n'%redgrad)



           # print(using('newton'+str(iNewton)))
           # if iNewton % 10 == 0 :
           #     outputVector2d(solver.u_adj, solver.u_adj.shape,'uadj'+str(iNewton)+'.dat')
           #     outputVector1d(solver.dt_adj,solver.dt_adj.shape, 'dtadj'+str(iNewton)+'.dat')
            
            #figure(1)
            #xlabel('time')
            #ylabel('residuum')
            #semilogy(solver.t[1:],sqrt((res**2).sum(1))[:])
   
        
        u.append((solver.u).copy())
        t.append((solver.t).copy())


    outputVector2d(solver.u,solver.u.shape,'u.dat')
    outputVector1d(solver.t,solver.t.shape, 't.dat')
    #outputVector2d(solver.u_adj, solver.u_adj.shape,'uadj.dat')
    #outputVector1d(solver.dt_adj,solver.dt_adj.shape, 'dtadj.dat')

    show(block=True)

#    u, t = array(u), array(t)
   
#    figure(figsize=(5,10))
#    contour4f(mus[:,newaxis] + t * 0, t, u[:,:,0], 501)
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
#blackboxfile.close()
redgradfile.close()
adjresfile.close()
#adjnextfile.close()
primresfile.close()
