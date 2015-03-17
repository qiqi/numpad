# Illustration of staggered grid (including ghost faces):
#
#          u_x[0,0]           u_x[1,0]           u_x[2,0]
#
#  u_y[0,0]   ----- u_y[1,0] --------- u_y[2,0] ---------
#             |                  |                  |
#             |                  |                  |
#          u_x[0,1]  p[0,0]   u_x[1,1]  p[1,0]   u_x[2,1]
#             |                  |                  |
#             |                  |                  |
#  u_y[0,1]   |---- u_y[1,1] ----+---- u_y[2,1] ----+----
#             |                  |                  |
#             |                  |                  |
#          u_x[0,2]  p[0,1]   u_x[1,2]  p[1,1]   u_x[2,2]
#             |                  |                  |
#
#                     ...                ...
#
#             |                  |                  |
#          u_x[0,N] p[0,N-1]  u_x[1,N] p[1,N-1]  u_x[2,N]
#             |                  |                  |
#             |                  |                  |
#  u_y[0,N]   ----- u_y[1,N] --------- u_y[2,N] ---------
#                                                        
#         u_x[0,N+1]         u_x[1,N+1]         u_x[2,N+1]

import sys
sys.path.append('../..')
from numpad import *

def make_array(u_and_p):
    p = u_and_p[:N*N].reshape([N, N])
    
    u_x = zeros([N+1, N+2])
    u_y = zeros([N+2, N+1])
    u_x[1:-1,1:-1] = u_and_p[N*N:N*(2*N-1)].reshape([N-1,N])
    u_y[1:-1,1:-1] = u_and_p[N*(2*N-1):].reshape([N,N-1])
    u_x[:,0] = 1-u_x[:,1]
    u_x[:,-1] = -u_x[:,-2]
    u_y[0,:] = -u_y[1,:]
    u_y[-1,:] = -u_y[-2,:]

    return p, u_x, u_y
    
def cavity(u_and_p, u_and_p0, dt, f=0):
    # extend velocity to ghost cells with boundary conditions
    p, u_x, u_y = make_array(0.5 * (u_and_p + u_and_p0))
    
    # compute cell center values
    u_x_c = 0.5 * (u_x[1:,1:-1] + u_x[:-1,1:-1])
    u_y_c = 0.5 * (u_y[1:-1,1:] + u_y[1:-1,:-1])
    u_xx_c = u_x_c**2
    u_yy_c = u_y_c**2

    # compute nodal values
    u_x_n = 0.5 * (u_x[:,1:] + u_x[:,:-1])
    u_y_n = 0.5 * (u_y[1:,:] + u_y[:-1,:])
    u_xy_n = u_x_n * u_y_n

    # compute derivatives
    du_xx_dx = (u_xx_c[1:,:] - u_xx_c[:-1,:]) /  dx
    du_yy_dy = (u_yy_c[:,1:] - u_yy_c[:,:-1]) /  dy

    du_xy_dx = (u_xy_n[1:,1:-1] - u_xy_n[:-1,1:-1]) / dx
    du_xy_dy = (u_xy_n[1:-1,1:] - u_xy_n[1:-1,:-1]) / dy
    
    dp_dx = (p[1:,:] - p[:-1,:]) / dx
    dp_dy = (p[:,1:] - p[:,:-1]) / dy
    
    # convective rhs
    conv_x = du_xx_dx + du_xy_dy + dp_dx
    conv_y = du_yy_dy + du_xy_dx + dp_dy
    
    # viscous rhs
    visc_x = (u_x[2:,1:-1] - 2 * u_x[1:-1,1:-1] + u_x[:-2,1:-1]) / dx**2 \
           + (u_x[1:-1,2:] - 2 * u_x[1:-1,1:-1] + u_x[1:-1,:-2]) / dy**2
    visc_y = (u_y[2:,1:-1] - 2 * u_y[1:-1,1:-1] + u_y[:-2,1:-1]) / dx**2 \
           + (u_y[1:-1,2:] - 2 * u_y[1:-1,1:-1] + u_y[1:-1,:-2]) / dy**2

    # time derivative
    u_x0 = u_and_p0[N*N:N*(2*N-1)].reshape([N-1,N])
    u_y0 = u_and_p0[N*(2*N-1):].reshape([N,N-1])

    du_x_dt = (u_x[1:-1,1:-1] - u_x0) / dt
    du_y_dt = (u_y[1:-1,1:-1] - u_y0) / dt
    
    # divergence free condition
    div_u = (u_x[1:,1:-1] - u_x[:-1,1:-1]) / dx \
          + (u_y[1:-1,1:] - u_y[1:-1,:-1]) / dy 
    div_u[0,0] = p[0,0]
    
    res = [div_u, du_x_dt + conv_x - visc_x / Re,
                  du_y_dt + conv_y - visc_y / Re]
    return hstack([ravel(r) for r in res]) - f

# ---------------------- time integration --------------------- #
N = 50
dx = dy = 1. / N
t, dt = 0, 0.1
Re = 10000

u_and_p_0 = zeros(N * (3 * N - 2))

for iTimeStep in range(1000):
    print('t = ', t)
    u_and_p = solve(cavity, u_and_p_0, args=(u_and_p_0, dt),
                    rel_tol=1E-6, abs_tol=1E-8)
    u_and_p.obliviate()

    res = cavity(u_and_p, u_and_p_0, dt)
    print(np.linalg.norm(value(res)))

    E_block = res.diff(u_and_p_0).tocsr()
    G_block = res.diff(u_and_p).tocsr()

    E_filename = 'E_block_{0:06d}.npz'.format(iTimeStep)
    G_filename = 'G_block_{0:06d}.npz'.format(iTimeStep)
    np.savez(E_filename, data=E_block.data, indices=E_block.indices, indptr=E_block.indptr)
    np.savez(G_filename, data=G_block.data, indices=G_block.indices, indptr=G_block.indptr)

    t += dt
    u_and_p_0 = u_and_p

# visualization
import numpy as np
x = (np.arange(N) + 0.5) * dx
y = (np.arange(N) + 0.5) * dy
p, u_x, u_y = make_array(u_and_p)
p = value(p)
u_x = 0.5 * value(u_x[1:,1:-1] + u_x[:-1,1:-1])
u_y = 0.5 * value(u_y[1:-1,1:] + u_y[1:-1,:-1])

from pylab import *
# streamplot(x, y, u_x.T, u_y.T, density=5)
contour(x, y, u_x.T, 50)
axis('scaled')
axis([0,1,0,1])
xlabel('x')
ylabel('y')
figure()
contour(x, y, u_y.T, 50)
axis('scaled')
axis([0,1,0,1])
xlabel('x')
ylabel('y')
# 
# savefig('cavity{0:d}.png'.format(Re))
