import time
import sys
from pylab import *
sys.path.append('../..')
from numpad import *

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
    mach2 = (u**2 + v**2) / c2
    rhot = rho * (1 + 0.2 * mach2)**2.5
    pt = p * (1 + 0.2 * mach2)**3.5

    d_rho = 1 - rho
    d_pt = pt_in - pt
    d_u = d_pt / (rho * (u + c))
    d_p = rho * c * d_u

    rho = rho + d_rho
    u = u + d_u
    p = p + d_p
    w[0,0,1:-1] = rho
    w[1,0,1:-1] = rho * u
    w[2,0,1:-1] = 0
    w[3,0,1:-1] = p / 0.4 + 0.5 * rho * u**2

    # outlet
    w[:,-1,1:-1] = w[:,-2,1:-1]
    rho, u, v, E, p = primative(w[:,-1,1:-1])
    w[3,-1,1:-1] = p_out / (1.4 - 1) + 0.5 * rho * (u**2 + v**2)

    # walls
    w[:,:,0] = w[:,:,1]
    nwall = geo.normal_j[:,:,0]
    nwall = hstack([nwall[:,:1], nwall, nwall[:,-1:]])
    rhoU_n = sum(w[1:3,:,0] * nwall, 0)
    w[1:3,:,0] -= 2 * rhoU_n * nwall

    w[:,:,-1] = w[:,:,-2]
    nwall = geo.normal_j[:,:,-1]
    nwall = hstack([nwall[:,:1], nwall, nwall[:,-1:]])
    rhoU_n = sum(w[1:3,:,-1] * nwall, 0)
    w[1:3,:,-1] -= 2 * rhoU_n * nwall

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

def sponge_flux(c_ext, w_ext, geo):
    ci = 0.5 * (c_ext[1:,1:-1] + c_ext[:-1,1:-1])
    cj = 0.5 * (c_ext[1:-1,1:] + c_ext[1:-1,:-1])

    a = geo.area
    ai = vstack([a[:1,:], (a[1:,:] + a[:-1,:]) / 2, a[-1:,:]])
    aj = hstack([a[:,:1], (a[:,1:] + a[:,:-1]) / 2, a[:,-1:]])

    wxx = (w_ext[:,2:,1:-1] + w_ext[:,:-2,1:-1] - 2 * w_ext[:,1:-1,1:-1]) / 3.
    wyy = (w_ext[:,1:-1,2:] + w_ext[:,1:-1,:-2] - 2 * w_ext[:,1:-1,1:-1]) / 3.
    # second order dissipation at boundary, fourth order in the interior
    Fi = -0.5 * ci * ai * (w_ext[:,1:,1:-1] - w_ext[:,:-1,1:-1])
    Fi[:,1:-1,:] = 0.5 * (ci * ai)[1:-1,:] * (wxx[:,1:,:] - wxx[:,:-1,:])
    Fj = -0.5 * cj * aj * (w_ext[:,1:-1,1:] - w_ext[:,1:-1,:-1])
    Fj[:,:,1:-1] = 0.5 * (cj * aj)[:,1:-1] * (wyy[:,:,1:] - wyy[:,:,:-1])
    return Fi, Fj

def euler_kec(w, w0, geo, dt):
    '''
    Kinetic energy conserving scheme with no numerical viscosity
    '''
    w_ext = extend(w, geo)
    rho, u, v, E, p = primative(w_ext)
    c = sqrt(1.4 * p / rho)
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
    # sponge
    Fi_s, Fj_s = sponge_flux(c, w_ext, geo)
    Fi += 0.5 * Fi_s
    Fj += 0.5 * Fj_s
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
def avg(a):
    return 0.25 * (a[1:,1:] + a[1:,:-1] + a[:-1,1:] + a[:-1,:-1])

def vis(w, geo):
    '''
    Visualize Mach number, non-dimensionalized stagnation and static pressure
    '''
    import numpy as np
    rho, u, v, E, p = primative(value(extend(w, geo)))
    x, y = value(geo.xy)
    xc, yc = value(geo.xyc)
    
    c2 = 1.4 * p / rho
    M = sqrt((u**2 + v**2) / c2)
    pt = p * (1 + 0.2 * M**2)**3.5

    subplot(1,2,1)
    # contourf(x, y, avg(M), 100)
    contourf(xc, yc, M[1:-1,1:-1], 100)
    colorbar()
    quiver(xc, yc, u[1:-1,1:-1], v[1:-1,1:-1])
    axis('scaled')
    xlabel('x')
    ylabel('y')
    title('Mach')
    draw()
    
    # subplot(2,2,2)
    # pt_frac = (pt - p_out) / (pt_in - p_out)
    # contourf(x, y, avg(pt_frac), 100)
    # colorbar()
    # axis('scaled')
    # xlabel('x')
    # ylabel('y')
    # title('pt')
    # draw()
    
    subplot(1,2,2)
    p_frac = (p - p_out) / (0.5 * rho[0,0] * u[0,0]**2)
    # contourf(x, y, avg(p_frac), 100)
    contourf(xc, yc, p_frac[1:-1,1:-1], 100)
    colorbar()
    axis('scaled')
    xlabel('x')
    ylabel('y')
    title('p')
    draw()

# ---------------------- time integration --------------------- #
geometry = 'bend'

if geometry == 'nozzle':
    Ni, Nj = 50, 20
    x = np.linspace(-20,20,Ni+1)
    y = np.linspace(-5, 5, Nj+1)
    a = np.ones(Ni+1)
    a[np.abs(x) < 10] = 1 - (1 + cos(x[np.abs(x) < 10] / 10 * np.pi)) * 0.1
    
    y, x = np.meshgrid(y, x)
    y *= a[:,np.newaxis]

elif geometry == 'bend':
    Ni, Nj = 90, 20
    # Ni, Nj = 200, 40
    theta = np.linspace(0, pi/2, Ni/3+1)
    r = 15 + 8 * np.sin(np.linspace(-np.pi/2, np.pi/2, Nj+1))
    r, theta = np.meshgrid(r, theta)
    x, y = r * np.sin(theta), r * np.cos(theta)

    dx = 24 / Nj
    y0 = y[0,:]
    y0, x0 = np.meshgrid(y0, dx * np.arange(-Ni/3, 0))

    x1 = x[-1,:]
    x1, y1 = np.meshgrid(x1, -dx * np.arange(1, 1 + Ni/3))
    
    x, y = np.vstack([x0, x, x1]), np.vstack([y0, y, y1])

geo = geo2d([x, y])

t, dt = 0, 0.1

pt_in = 1.05E5
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

    print(adarray_count(), adstate_count())

    # if i % 10 == 0:
    #     vis(w, geo)

print('Final, t = inf')
dt = np.inf
w = solve(euler_kec, w0, args=(w0, geo, dt), rel_tol=1E-8, abs_tol=1E-6)
figure(figsize=(30,10))
vis(w, geo)

rho, u, v, E, p = [value(pi) for pi in primative(extend(w, geo))]

