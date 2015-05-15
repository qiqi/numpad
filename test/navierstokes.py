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
    mach2 = u**2 / c2
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
    p = p_out
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

def grad_dual(phi, geo):
    '''
    Gradient on the nodes assuming zero boundary conditions
    '''
    dxy_i = 0.5 * (geo.dxy_i[:,1:,:] + geo.dxy_i[:,:-1,:])
    dxy_j = 0.5 * (geo.dxy_j[:,:,1:] + geo.dxy_j[:,:,:-1])
    phi_i = array([phi * dxy_i[1], -phi * dxy_i[0]])
    phi_j = array([-phi * dxy_j[1], phi * dxy_j[0]])

    grad_phi = zeros(geo.xy.shape)
    grad_phi[:,:-1,:-1] += phi_i + phi_j
    grad_phi[:,1:,:-1] += -phi_i + phi_j
    grad_phi[:,:-1,1:] += phi_i - phi_j
    grad_phi[:,1:,1:] += -phi_i - phi_j

    area = zeros(geo.xy.shape[1:])
    area[:-1,:-1] += geo.area
    area[1:,:-1] += geo.area
    area[:-1,1:] += geo.area
    area[1:,1:] += geo.area

    return 2 * grad_phi / area

def ns_flux(rho, u, v, E, p, grad_u):
    # viscous stress
    dudx, dudy, dvdx, dvdy = grad_u
    mu_t = 0 # blabla
    sigma_xx = (mu + mu_t) * (2 * dudx - 2./3 * (dudx + dvdy))
    sigma_yy = (mu + mu_t) * (2 * dvdy - 2./3 * (dudx + dvdy))
    sigma_xy = (mu + mu_t) * (dudy + dvdx)

    F = array([rho * u, rho * u**2 + p - sigma_xx,
                        rho * u * v    - sigma_xy,
                        u * (E + p)    - sigma_xx * u - sigma_xy * v])
    G = array([rho * v, rho * u * v    - sigma_xy,
                        rho * v**2 + p - sigma_yy,
                        v * (E + p)    - sigma_xy * u - sigma_yy * v])
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

def ns_kec(w, w0, geo, dt):
    '''
    Kinetic energy conserving scheme with no numerical viscosity
    '''
    w_ext = extend(w, geo)
    rho, u, v, E, p = primative(w_ext)
    c = sqrt(1.4 * p / rho)
    # velocity gradient on nodes
    dudx, dudy = grad_dual(u[1:-1,1:-1], geo)
    dvdx, dvdy = grad_dual(v[1:-1,1:-1], geo)
    duv_dxy = array([dudx, dudy, dvdx, dvdy])
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
    # interface strain rate averged from dual mesh (nodal) values
    duv_dxy_i = 0.5 * (duv_dxy[:,:,1:] + duv_dxy[:,:,:-1])
    duv_dxy_j = 0.5 * (duv_dxy[:,1:,:] + duv_dxy[:,:-1,:])
    # inlet and outlet have no viscous stress
    duv_dxy_i[:,[0,-1],:] = 0
    # interface flux
    f_i, g_i = ns_flux(rho_i, u_i, v_i, E_i, p_i, duv_dxy_i)
    f_j, g_j = ns_flux(rho_j, u_j, v_j, E_j, p_j, duv_dxy_j)
    Fi = + f_i * geo.dxy_i[1] - g_i * geo.dxy_i[0]
    Fj = - f_j * geo.dxy_j[1] + g_j * geo.dxy_j[0]
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
def vis(w, geo):
    '''
    Visualize Mach number, non-dimensionalized stagnation and static pressure
    '''
    def avg(a):
        return 0.25 * (a[1:,1:] + a[1:,:-1] + a[:-1,1:] + a[:-1,:-1])

    import numpy as np
    rho, u, v, E, p = primative(value(extend(w, geo)))
    x, y = value(geo.xy)
    xc, yc = value(geo.xyc)
    
    c2 = 1.4 * p / rho
    M = sqrt((u**2 + v**2) / c2)
    pt = p * (1 + 0.2 * M**2)**3.5

    subplot(2,2,1)
    contourf(x, y, avg(M), 100)
    colorbar()
    quiver(xc, yc, u[1:-1,1:-1], v[1:-1,1:-1])
    axis('scaled')
    xlabel('x')
    ylabel('y')
    title('Mach')
    draw()
    
    subplot(2,2,2)
    pt_frac = (pt - p_out) / (pt_in - p_out)
    contourf(x, y, avg(pt_frac), 100)
    colorbar()
    axis('scaled')
    xlabel('x')
    ylabel('y')
    title('pt')
    draw()
    
    subplot(2,2,3)
    p_frac = (p - p_out) / (pt_in - p_out)
    contourf(x, y, avg(p_frac), 100)
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
    # Ni, Nj = 100, 40
    x = linspace(-15,25,Ni+1)
    y = sin(linspace(-np.pi/2, np.pi/2, Nj+1))
    a = ones(Ni+1)
    a[np.abs(x) < 10] = 1 - (1 + cos(x[np.abs(x) < 10] / 10 * np.pi)) * 0.2
    
    y, x = np.meshgrid(y, x)
    y *= 5 * a[:,np.newaxis]

elif geometry == 'bend':
    Ni, Nj = 80, 20
    # Ni, Nj = 200, 40
    theta = linspace(0, pi, Ni/2+1)
    r = 15 + 5 * sin(linspace(-np.pi/2, np.pi/2, Nj+1))
    r, theta = meshgrid(r, theta)
    x, y = r * sin(theta), r * cos(theta)

    dx = 15 * 2 * pi / Ni
    y0, y1 = y[0,:], y[-1,:]
    y0, x0 = meshgrid(y0, dx * arange(-Ni/4, 0))
    y1, x1 = meshgrid(y1, -dx * arange(1, 1 + Ni/4))
    
    x, y = vstack([x0, x, x1]), vstack([y0, y, y1])

np.save('geo.npy', value(array([x, y])))
geo = geo2d([x, y])

t, dt = 0, 1./Nj

pt_in = 1.2E5
p_out = 1E5
mu = 1

w = zeros([4, Ni, Nj])
w[0] = 1
w[3] = 1E5 / (1.4 - 1)

w0 = ravel(w)

for i in range(100):
    print('i = ', i, 't = ', t)
    w = solve(ns_kec, w0, args=(w0, geo, dt), rel_tol=1E-6, abs_tol=1E-4)
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
    #     show(block=True)

print('Final, t = inf')
dt = np.inf
w = solve(ns_kec, w0, args=(w0, geo, dt), rel_tol=1E-6, abs_tol=1E-4)
figure(figsize=(30,10))
vis(w, geo)
savefig('navierstokes-{0}.png'.format(geometry))

open('graph.dot', 'wt').write(dot(w))

show(block=True)
