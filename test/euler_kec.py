import time
import sys
sys.path.append('..')
from adarray import *
from adsolve import *

def extend(w_interior, geo):
    '''
    Extend the conservative variables into ghost cells using boundary condition
    '''
    w = zeros([4, Ni+2, Nj+2])
    w[:,1:-1,1:-1] = w_interior.reshape([4, Ni, Nj])

    # inlet
    rho, u, v, E, p = primative(w[:,1,1:-1])
    c2 = 1.4 * p / rho
    c = sqrt(c2)
    mach2 = u**2 / c2
    rhot = rho * (1 + 0.2 * mach2)**2.5
    pt = p * (1 + 0.2 * mach2)**3.5

    d_rho = 1 - rho
    d_pt = pt_in - pt
    d_u = d_pt / (rho * (u + c))
    d_p = rho * c * d_u

    relax = 0.5
    rho = rho + relax * d_rho
    u = u + relax * d_u
    p = p + relax * d_p
    w[0,0,1:-1] = rho
    w[1,0,1:-1] = rho * u
    w[2,0,1:-1] = 0
    w[3,0,1:-1] = p / 0.4 + 0.5 * rho * u**2

    # outlet
    w[:,-1,1:-1] = w[:,-2,1:-1]
    rho, u, v, E, p = primative(w[:,-1,1:-1])
    p = relax * p_out + (1 - relax) * p
    w[3,-1,1:-1] = p / (1.4 - 1) + 0.5 * rho * (u**2 + v**2)

    # walls
    w[:,:,0] = w[:,:,1]
    rhoU_n = sum(w[1:3,1:-1,0] * geo.normal_j[:,:,0], 0)
    w[1:3,1:-1,0] -= 2 * rhoU_n * geo.normal_j[:,:,0]

    w[:,:,-1] = w[:,:,-2]
    rhoU_n = sum(w[1:3,1:-1,-1] * geo.normal_j[:,:,-1], 0)
    w[1:3,1:-1,-1] -= 2 * rhoU_n * geo.normal_j[:,:,-1]

    return w
    
def primative(w):
    '''
    Transform conservative variables into primative ones
    '''
    rho = w[0]
    u = w[1] / rho
    v = w[2] / rho
    E = w[3]
    p = 0.4 * (E - 0.5 * (u * w[1] + v * w[2]))
    return rho, u, v, E, p

def euler_flux(rho, u, v, E, p):
    F = array([rho*u, rho*u**2 + p, rho*u*v, u*(E + p)])
    G = array([rho*v, rho*u*v, rho*v**2 + p, v*(E + p)])
    return F, G

def euler_kec(w, w0, geo, dt):
    '''
    Kinetic energy conserving scheme with no numerical viscosity
    '''
    w_ext = extend(w, geo)
    rho, u, v, E, p = primative(w_ext)
    # interface average
    rho_i = 0.5 * (rho[1:,1:-1] + rho[:-1,1:-1])
    rho_j = 0.5 * (rho[1:-1,1:] + rho[1:-1,:-1])
    u_i = 0.5 * (u[1:,1:-1] + u[:-1,1:-1])
    u_j = 0.5 * (u[1:-1,1:] + u[1:-1,:-1])
    v_i = 0.5 * (v[1:,1:-1] + v[:-1,1:-1])
    v_j = 0.5 * (v[1:-1,1:] + v[1:-1,:-1])
    E_i = 0.5 * (E[1:,1:-1] + E[:-1,1:-1])
    E_j = 0.5 * (E[1:-1,1:] + E[1:-1,:-1])
    p_i = 0.5 * (p[1:,1:-1] + p[:-1,1:-1])
    p_j = 0.5 * (p[1:-1,1:] + p[1:-1,:-1])
    # interface flux
    F_i, G_i = euler_flux(rho_i, u_i, v_i, E_i, p_i)
    F_j, G_j = euler_flux(rho_j, u_j, v_j, E_j, p_j)
    Fi = + F_i * geo.dxy_i[1] - G_i * geo.dxy_i[0]
    Fj = - F_j * geo.dxy_j[1] + G_j * geo.dxy_j[0]
    # residual
    divF = (Fi[:,1:,:] - Fi[:,:-1,:] + Fj[:,:,1:] - Fj[:,:,:-1]) / geo.area
    return (w - w0) / dt + ravel(divF)


# -------------------------- geometry ------------------------- #
class geo2d:
    def __init__(self, xy):
        xy = array(xy)
        self.xy = xy
        self.xyc = (xy[:,1:,1:]  + xy[:,:-1,1:] + \
                    xy[:,1:,:-1] + xy[:,:-1,:-1]) / 4

        self.dxy_i = xy[:,:,1:] - xy[:,:,:-1]
        self.dxy_j = xy[:,1:,:] - xy[:,:-1,:]

        self.L_j = sqrt(self.dxy_j[0]**2 + self.dxy_j[1]**2)
        self.normal_j = array([self.dxy_j[1] / self.L_j,
                              -self.dxy_j[0] / self.L_j])

        self.area = self.tri_area(self.dxy_i[:,:-1,:], self.dxy_j[:,:,1:]) \
                  + self.tri_area(self.dxy_i[:,1:,:], self.dxy_j[:,:,:-1]) \

    def tri_area(self, xy0, xy1):
        return 0.5 * (xy0[1] * xy1[0] - xy0[0] * xy1[1])
        

# ----------------------- visualization --------------------------- #
def vis(w, geo):
    '''
    Visualize Mach number, non-dimensionalized stagnation and static pressure
    '''
    import numpy as np
    rho, u, v, E, p = primative(base(extend(w, geo)[:,1:-1,1:-1]))
    x, y = base(geo.xyc)
    
    from pylab import figure, subplot, contourf, colorbar, \
            quiver, axis, xlabel, ylabel, draw, show, title

    c2 = 1.4 * p / rho
    M = sqrt((u**2 + v**2) / c2)
    pt = p * (1 + 0.2 * M**2)**3.5

    figure()
    subplot(2,2,1)
    contourf(x, y, M, 100)
    colorbar()
    quiver(x, y, u, v)
    axis('scaled')
    xlabel('x')
    ylabel('y')
    title('Mach')
    draw()
    
    subplot(2,2,2)
    pt_frac = (pt - p_out) / (pt_in - p_out)
    contourf(x, y, pt_frac, 100)
    colorbar()
    axis('scaled')
    xlabel('x')
    ylabel('y')
    title('pt')
    draw()
    
    subplot(2,2,3)
    p_frac = (p - p_out) / (pt_in - p_out)
    contourf(x, y, p_frac, 100)
    colorbar()
    axis('scaled')
    xlabel('x')
    ylabel('y')
    title('p')
    draw()

    show(block=True)

# ---------------------- time integration --------------------- #
Ni, Nj = 50, 20
x = np.linspace(-20,20,Ni+1)
y = np.linspace(-5, 5, Nj+1)
a = np.ones(Ni+1)
a[np.abs(x) < 10] = 1 - (1 + cos(x[np.abs(x) < 10] / 10 * np.pi)) * 0.1

y, x = np.meshgrid(y, x)
y *= a[:,np.newaxis]

geo = geo2d([x, y])

t, dt = 0, 0.1

pt_in = 1.2E5
p_out = 1E5

w = zeros([4, Ni, Nj])
w[0] = 1
w[3] = 1E5 / (1.4 - 1)

w0 = ravel(w)

for i in range(100):
    print('i = ', i, 't = ', t)
    w = solve(euler_kec, w0, args=(w0, geo, dt), rel_tol=1E-8, abs_tol=1E-6)
    if w._n_Newton == 1:
        break
    elif w._n_Newton < 5:
        w0 = w
        dt *= 2
    elif w._n_Newton < 10:
        w0 = w
    else:
        dt *= 0.5
        continue
    t += dt
    w0.obliviate()

    # if i % 10 == 0:
    #     vis(w, geo)

print('Final, t = inf')
dt = np.inf
w = solve(euler_kec, w0, args=(w0, geo, dt), rel_tol=1E-8, abs_tol=1E-6)
vis(w, geo)
