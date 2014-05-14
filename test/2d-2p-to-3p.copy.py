from time import sleep
import matplotlib
matplotlib.interactive(True)

from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from numpy import *
import scipy.sparse
import scipy.sparse.linalg as splinalg

import matplotlib.pyplot as plt
import sys
sys.path.append('/home/qiqi/git')
from numpad import *

class ReservoirState:
    def __init__(self, po_sw_sg, is_there_gas, res, convert_s_g_r_s=False):
        '''
        p_o: pressure
        s_w: saturation
        res: reservoir description
        '''
        assert is_there_gas.shape == (res.Nx, res.Ny)
        self.is_there_gas = is_there_gas
        po_sw_sg = po_sw_sg.reshape([3, res.Nx, res.Ny])
        self.p_o = po_sw_sg[0].copy()
        self.s_w = po_sw_sg[1].copy()
        # If there is free gas, set s_g as unknown. Let r_s depends on p_o
        self.s_g = po_sw_sg[2].copy()
        self.r_s = self.p_o * 0.058
        # Is there is no free gas, set r_s as unknonw. s_g must be 0.
        self.r_s[~is_there_gas] = po_sw_sg[2][~is_there_gas]
        self.s_g[~is_there_gas] = 0
        self.res = res
        # I don't believe the given is_there_gas
        if convert_s_g_r_s:
            s_g_to_r_s = logical_and(is_there_gas, base(self.s_g) < 0)
            self.s_g[s_g_to_r_s] = 0
            self.r_s[s_g_to_r_s] = self.p_o[s_g_to_r_s] * 0.058
            r_s_to_s_g = logical_and(~is_there_gas, base(self.r_s) >= base(self.p_o) * 0.058)
            self.r_s[r_s_to_s_g] = self.p_o[r_s_to_s_g] * 0.058
            self.s_g[r_s_to_s_g] = 1E-6  # so that it is classified correctly in the next line
            self.is_there_gas = (base(self.s_g) > 0)
        self.update_states(res)

    def po_sw_sg(self):
        # provide initial guess for the unknowns of the next time step
        po_sw_sg = array([self.p_o, self.s_w, self.s_g])
        no_gas = ~self.is_there_gas
        po_sw_sg[2][no_gas] = self.r_s[no_gas]
        return ravel(po_sw_sg)

    def total(self):
        res = self.res
        volume = res.dx[:,newaxis] * res.dy[newaxis,:] * res.dz
        volume_o = volume * self.phi * self.s_o
        volume_w = volume * self.phi * self.s_w
        volume_g = volume * self.phi * self.s_g
        total_water = (Water.density(res,self) * volume_w).sum()
        total_oil = (Oil.density(res,self) * volume_o).sum()
        total_gas = (Gas.density(res,self) * volume_g).sum()
        return total_oil, total_water, total_gas

    def total_rate(self):
        total_rate_w = (Water.density(res,self) * self.q_w).sum() * 5.615
        total_rate_o = (Oil.density(res,self) * self.q_o).sum() * 5.615
        total_rate_g = (Gas.density(res,self) * self.q_o).sum() * 5.615
        return total_rate_o, total_rate_w, total_rate_g

    def gradient(self, p):
        res = self.res
        dx_cc = 0.5 * (res.dx[1:] + res.dx[:-1])[:,newaxis]
        dy_cc = 0.5 * (res.dy[1:] + res.dy[:-1])[newaxis,:]

        u, v = zeros((2, res.Nx, res.Ny))

        u[1:-1,:] = (p[2:,:] - p[:-2,:]) / (dx_cc[1:,:] + dx_cc[:-1,:])
        u[0,:] = (p[1,:] - p[0,:]) / dx_cc[0,:]
        u[-1,:] = (p[-1,:] - p[-2,:]) / dx_cc[-1,:]

        v[:,1:-1] = (p[:,2:] - p[:,:-2]) / (dy_cc[:,1:] + dy_cc[:,:-1])
        v[:,0] = (p[:,1] - p[:,0]) / dy_cc[:,0]
        v[:,-1] = (p[:,-1] - p[:,-2]) / dy_cc[:,-1]

        return u, v

    def water_velocity(self):
        u, v = self.gradient(self.p_w)
        u *= -res.k_cc[:,:,0] * self.lambda_cc_w
        v *= -res.k_cc[:,:,1] * self.lambda_cc_w
        return u, v

    def oil_velocity(self):
        u, v = self.gradient(self.p_o)
        u *= -res.k_cc[:,:,0] * self.lambda_cc_o
        v *= -res.k_cc[:,:,1] * self.lambda_cc_o
        return u, v

    def gas_velocity(self):
        u, v = self.gradient(self.p_g)
        u *= -res.k_cc[:,:,0] * self.lambda_cc_g
        v *= -res.k_cc[:,:,1] * self.lambda_cc_g
        return u, v

    def update_states(self, res):
        self.s_o = 1. - self.s_w - self.s_g

        self.p_cow = 0 #(1. - self.s_w) / (self.s_w - res.s_wc) * res.p_cow_mult
        self.p_w = self.p_o - self.p_cow

        self.p_cog = 0 
        self.p_g = self.p_o - self.p_cog
       
        # self.rho_o = res.rho_o_ref * exp(res.c_o * (self.p_o - res.p_ref)) 
        # self.rho_w = res.rho_w_ref * exp(res.c_w * (self.p_w - res.p_ref)) 
        # self.rho_g = Gas.density(res, self)
        # self.pot_o = self.p_o - self.rho_o / 144. * res.z
        # self.pot_w = self.p_w - self.rho_w / 144. * res.z
        # self.pot_g = self.p_g - self.rho_g / 144. * res.z

        # self.kr_w = res.kr_w_func(self.s_w)
        # self.kr_o = res.kr_o_func(self.s_w)

        self.lambda_cc_w = Water.relative_permeability(res, self) / Water.viscosity(res, self)
        self.lambda_cc_o = Oil.relative_permeability(res, self) / Oil.viscosity(res, self)
        self.lambda_cc_g = Gas.relative_permeability(res, self) / Gas.viscosity(res, self)
    
        # HERE
        # self.lambda_wx = self.lambda_cc_w[:-1] * gt_smooth(self.pot_w[:-1], self.pot_w[1:]) \
        #                + self.lambda_cc_w[1:] *  lt_smooth(self.pot_w[:-1], self.pot_w[1:])
        # self.lambda_wy = self.lambda_cc_w[:,:-1] * gt_smooth(self.pot_w[:,:-1], self.pot_w[:,1:]) \
        #                + self.lambda_cc_w[:,1:]  * lt_smooth(self.pot_w[:,:-1], self.pot_w[:,1:])
        # self.lambda_ox = self.lambda_cc_o[:-1] * gt_smooth(self.pot_o[:-1] , self.pot_o[1:]) \
        #                + self.lambda_cc_o[1:]  * lt_smooth(self.pot_o[:-1] , self.pot_o[1:])
        # self.lambda_oy = self.lambda_cc_o[:,:-1] * gt_smooth(self.pot_o[:,:-1] , self.pot_o[:,1:]) \
        #                + self.lambda_cc_o[:,1:]  * lt_smooth(self.pot_o[:,:-1] , self.pot_o[:,1:])
        # self.lambda_gx = self.lambda_cc_g[:-1] * gt_smooth(self.pot_g[:-1] , self.pot_g[1:]) \
        #                + self.lambda_cc_g[1:]  * lt_smooth(self.pot_g[:-1] , self.pot_g[1:])
        # self.lambda_gy = self.lambda_cc_g[:,:-1] * gt_smooth(self.pot_g[:,:-1] , self.pot_g[:,1:]) \
        #                + self.lambda_cc_g[:,1:]  * lt_smooth(self.pot_g[:,:-1] , self.pot_g[:,1:])
    
        self.phi = res.phi_ref * exp(res.c_r * (self.p_o - res.p_ref))
        self.Vp = res.dx[:,newaxis] * res.dy[newaxis,:] * res.dz * self.phi / 5.615
        self.Vp[0,:] *= 100
        self.Vp[-1,:] *= 100
        self.Vp[:,0] *= 100
        self.Vp[:,-1] *= 100

        # f_ox = self.lambda_ox / (self.lambda_ox + self.lambda_wx)
        # f_wx = self.lambda_wx / (self.lambda_ox + self.lambda_wx)
        # f_oy = self.lambda_oy / (self.lambda_oy + self.lambda_wy)
        # f_wy = self.lambda_wy / (self.lambda_oy + self.lambda_wy)

        # Here
        f_o_0 = zeros([res.Nx, res.Ny])
        f_w_0 = zeros([res.Nx, res.Ny])
        f_g_0 = zeros([res.Nx, res.Ny])
        f_o_0[res.w_type == 1] = (self.lambda_cc_o * res.k_cc.sum(2) * res.dx[:,newaxis])[res.w_type == 1]
        f_w_0[res.w_type == 1] = (self.lambda_cc_w * res.k_cc.sum(2) * res.dx[:,newaxis])[res.w_type == 1]
        f_g_0[res.w_type == 1] = (self.lambda_cc_g * res.k_cc.sum(2) * res.dx[:,newaxis])[res.w_type == 1]

        f_o = f_o_0 / (f_o_0 + f_w_0 + f_g_0).sum()
        f_w = f_w_0 / (f_o_0 + f_w_0 + f_g_0).sum()
        f_g = f_g_0 / (f_o_0 + f_w_0 + f_g_0).sum()

        # wells
        self.q_o = zeros([res.Nx, res.Ny])
        self.q_w = zeros([res.Nx, res.Ny])
        self.q_g = zeros([res.Nx, res.Ny])
        self.q_w[res.w_type == 2] = (f_w * res.q_inj)[res.w_type == 2]

        q_o_max = res.well_index * self.lambda_cc_o * (self.p_o - res.p_bp)
        q_w_max = res.well_index * self.lambda_cc_w * (self.p_o - res.p_bp)
        q_g_max = res.well_index * self.lambda_cc_g * (self.p_o - res.p_bp)
        q_o_prod = maximum_smooth(-q_o_max, f_o * res.q_pro)
        q_w_prod = maximum_smooth(-q_w_max, f_w * res.q_pro)
        q_g_prod = maximum_smooth(-q_g_max, f_g * res.q_pro)
        # q_o_prod = -q_o_max
        # q_w_prod = -q_w_max
        # q_g_prod = -q_g_max
        self.q_o[res.w_type == 1] = q_o_prod[res.w_type == 1]
        self.q_w[res.w_type == 1] = q_w_prod[res.w_type == 1]
        self.q_g[res.w_type == 1] = q_g_prod[res.w_type == 1]

params = {
    's_wc' : 0.1,
    'Nx' : 50,
    'Ny' : 50,
    'Lx' : 2.e3,
    'Ly' : 2.e3,
    'Lz' : 50,
    'dz' : 30.,
    'dzdx' : 0., # z slop in x direction 
    'dzdy' : 0., # z slop in y direction 
    'z0' : 7.e3,
    'p_ref' : 4.5e3,
    'phi_ref' : 0.2,
    'c_o' : 10e-6,
    'c_w' : 5e-5,
    'c_r' : 3e-6,
    'mu_o' : 1.,
    'mu_w' : 1.,
    'mu_g' : 0.01,
    'kx_norm' : array([200, 1, 50, 100]),
    'ky_norm' : array([200, 1, 50,  10]),
    'rho_w_ref' : 62.5,
    'rho_o_ref' : 20.,
    'p_cow_mult' : 10., # p_cow = 0 - without capilary pressure
    'q_inj' : 1000,
    'q_pro' : -18000,
    'p_bp' : 1200,
    'rw' : 0.5,
    'temp' : 250, # K
    'r_gas' : 10.73,
    'gas_gravity' : 0.8,
    'rs_ini' : 148
}

class Reservoir:
    def __init__(self, params):
        for key, value in params.items():
            self.__dict__[key] = value

        # generate mesh
        self.x_ci = linspace(0, self.Lx, self.Nx+1)
        self.x_cc = (self.x_ci[1:] + self.x_ci[:-1]) / 2.
        self.dx = self.x_ci[1:] - self.x_ci[:-1]
        self.y_ci = linspace(0, self.Ly, self.Ny+1)
        self.y_cc = (self.y_ci[1:] + self.y_ci[:-1]) / 2.
        self.dy = self.y_ci[1:] - self.y_ci[:-1]
        self.z = zeros([self.Nx, self.Ny])
        # HERE
        x_center = self.Lx / 2
        y_center = self.Ly / 2
        c = -self.Lz / (self.Lx/2)**2
        
        # self.z = c * ((self.x_cc[:,newaxis] - x_center)**2
        #             + (self.y_cc[newaxis,:] - y_center)**2)
        # self.z = self.z0 + self.dzdx * self.x_cc[:,newaxis]\
        #        + self.dzdy * self.y_cc[newaxis,:]
        # self.z_w = hstack([self.z[0], self.z[:-1]])
        # self.z_e = hstack([self.z[1:], self.z[-1]])

        # compute permeability and transmissibility
        # anisotropic permeability
#         self.k_cc = zeros([self.Nx, self.Ny, 2, 2]) # cell center k (Nx, Ny, 2, 2)
#         self.k_cc[:,:,0,0] = self.k_norm   # set diagonal term k (Nx, Ny, 2, 2)
#         self.k_cc[:,:,1,1] = self.k_norm   # set diagonal term k (Nx, Ny, 2, 2)
#         # cell interface kx (Nx-1)
#         self.kx_ci = (self.dx[:-1,newaxis,newaxis,newaxis] + self.dx[1:,newaxis,newaxis,newaxis]) \
#                    * self.k_cc[:-1] * self.k_cc[1:] \
#             / (self.dx[:-1,newaxis,newaxis,newaxis] * self.k_cc[1:] + self.dx[1:,newaxis,newaxis,newaxis] * self.k_cc[:-1])
#         self.ky_ci = (self.dy[:-1,newaxis,newaxis,newaxis] + self.dy[1:,newaxis,newaxis,newaxis]) \
#                    * self.k_cc[:,:-1] * self.k_cc[:,1:] \
#             / (self.dy[:-1,newaxis,newaxis,newaxis] * self.k_cc[:,1:] + self.dy[1:,newaxis,newaxis,newaxis] * self.k_cc[:,:-1])

        # compute permeability and transmissibility
        self.k_cc = ones([self.Nx, self.Ny, 2]) * 200
        # self.k_cc = zeros([self.Nx, self.Ny, 2])
        # self.k_cc[:int(self.Nx/2), :int(self.Ny/2), 0] = self.kx_norm[0]
        # self.k_cc[int(self.Nx/2):, :int(self.Ny/2), 0] = self.kx_norm[1]
        # self.k_cc[:int(self.Nx/2), int(self.Ny/2):, 0] = self.kx_norm[2]
        # self.k_cc[int(self.Nx/2):, int(self.Ny/2):, 0] = self.kx_norm[3]
        # self.k_cc[:int(self.Nx/2), :int(self.Ny/2), 1] = self.ky_norm[0]
        # self.k_cc[int(self.Nx/2):, :int(self.Ny/2), 1] = self.ky_norm[1]
        # self.k_cc[:int(self.Nx/2), int(self.Ny/2):, 1] = self.ky_norm[2]
        # self.k_cc[int(self.Nx/2):, int(self.Ny/2):, 1] = self.ky_norm[3]
        # cell interface kx (Nx-1,Ny)
        self.kx_ci = (self.dx[:-1,newaxis] + self.dx[1:,newaxis]) \
                   * self.k_cc[:-1,:,0] * self.k_cc[1:,:,0] \
            / (self.dx[:-1,newaxis] * self.k_cc[1:,:,0] + \
               self.dx[1:,newaxis] * self.k_cc[:-1,:,0])
        # cell interface ky (Nx,Ny-1)
        self.ky_ci = (self.dy[newaxis,:-1] + self.dy[newaxis,1:]) \
                   * self.k_cc[:,:-1,1] * self.k_cc[:,1:,1] \
            / (self.dy[newaxis,:-1] * self.k_cc[:,1:,1] \
             + self.dy[newaxis,1:] * self.k_cc[:,:-1,1])
        self.T_Rx = zeros([self.Nx+1,self.Ny])
        self.T_Ry = zeros([self.Nx,self.Ny+1])
        self.T_Rx[1:-1,:] = 1.127e-3 * self.kx_ci * self.dz * self.dy[newaxis,:] \
                       / (self.x_cc[1:,newaxis] - self.x_cc[:-1,newaxis])
        self.T_Ry[:,1:-1] = 1.127e-3 * self.ky_ci * self.dz * self.dx[:,newaxis] \
                       / (self.y_cc[newaxis,1:] - self.y_cc[newaxis,:-1])

        # Set  well information ########
        #w_num = 2    # well number
        self.w_type = np.zeros([self.Nx,self.Ny]) # well type: 1- total rate specified, 
                                             #            2- water rate specified,
                                             #            3- bh pressure specified
        self.q_rate = zeros([self.Nx,self.Ny]) # rate of well, "+" - injection, 
                        # bh pressure well: specified bottom hole pressuer
        # self.w_type[int(self.Nx*3/10),int(self.Ny*3/10)] = 1
        self.w_type[int(self.Nx/2),int(self.Ny/2)] = 1
        # self.w_type[int(self.Nx*7/10),int(self.Ny*7/10)] = 1
        # self.w_type[int(self.Nx*7/10),int(self.Ny*7/10)] = 5
        # self.w_type[0,0] = 2
        # self.w_type[0,-1] = 2
        # self.w_type[-1,0] = 2
        # self.w_type[-1,-1] = 2
        # self.q_rate[0,0] = self.q_inj
        # self.w_type[-1,-1] = 1
        # self.w_type[0,-1] = 1
        # self.q_rate[-1,-1] = -self.q_pro

        # load relative permeability table
        # sw_table, kr_w_table, kr_o_table = loadtxt('kr_sw.dat', skiprows=1).T
        # self.kr_w_func = interp(sw_table, kr_w_table)
        # self.kr_o_func = interp(sw_table, kr_o_table)

        self.well_index = 7.06e-3 * sqrt(self.k_cc[:,:,0] * self.k_cc[:,:,1]) \
                * self.dz / log(0.2 * self.dx[:,newaxis] / self.rw)

        # create well
        # self.well = Well(self, int(self.Nx/2), int(self.Ny/2), self.q_pro)

    @staticmethod
    def central_avg(rho):
        rho_ix = 0.5 * (rho[:-1,:] + rho[1:,:])
        rho_iy = 0.5 * (rho[:,:-1] + rho[:,1:])
        return rho_ix, rho_iy

    @staticmethod
    def upstream_avg(rho, potential):
        rho_ix = rho[:-1,:] * gt_smooth(potential[:-1,:], potential[1:,:]) \
               + rho[1:,:]  * lt_smooth(potential[:-1,:], potential[1:,:])
        rho_iy = rho[:,:-1] * gt_smooth(potential[:,:-1], potential[:,1:]) \
               + rho[:,1:]  * lt_smooth(potential[:,:-1], potential[:,1:])
        return rho_ix, rho_iy


class Oil:
    @staticmethod
    def density(res, state):
        return res.rho_o_ref * exp(res.c_o * (state.p_o - res.p_ref)) 

    @staticmethod
    def potential(res, state):
        return state.p_o - Oil.density(res, state) / 144. * res.z

    @staticmethod
    def formation_volume_factor(res, state):
        P_b = state.r_s / 0.058 
        # P_b = res.p_bp  # fixed bubble point
        B_ob = 1.391E-4 * P_b + 1.1282
        B_o = B_ob * (1 - res.c_o * (state.p_o - P_b))
        # B_o = 1.0
        return B_o

    @staticmethod
    def viscosity(res, state):
        P_b = state.r_s / 0.058 
        # P_b = res.p_bp 
        C_mu = 4.6E-5
        mu_ob = -9.140E-5 * P_b + 0.9351
        mu_o = mu_ob * (1 + C_mu * (state.p_o - P_b))
        # mu_o = res.mu_o
        return mu_o

    @staticmethod
    def relative_permeability(res, state):
        s_w, s_o, s_g = state.s_w, state.s_o, state.s_g
        # k_o = ((1 - s_w)**2 + s_w**2) - s_w**2
        k_o = ((1 - s_w)**2 + s_w**2) * ((1 - s_g)**2 + s_g**2) - s_w**2 - s_g**2
        return k_o

    @staticmethod
    def volume_flux(res, state):
        rho_ix, rho_iy = Reservoir.central_avg(Oil.density(res, state))
        pot_corr_east = state.p_o[1:,:] - rho_ix / 144. * res.z[1:,:]
        pot_corr_west = state.p_o[:-1,:] - rho_ix / 144. * res.z[:-1,:]
        pot_corr_south = state.p_o[:,1:] - rho_iy / 144. * res.z[:,1:]
        pot_corr_north = state.p_o[:,:-1] - rho_iy / 144. * res.z[:,:-1]

        kr_x, kr_y = Reservoir.upstream_avg(Oil.relative_permeability(res, state),
                                            Oil.potential(res, state))
        B_x, B_y = Reservoir.central_avg(Oil.formation_volume_factor(res, state))
        # B_x, B_y = 1., 1.
        mu_x, mu_y = Reservoir.central_avg(Oil.viscosity(res, state))
        # mu_x, mu_y = res.mu_o, res.mu_o
        flux_x = res.T_Rx[1:-1,:] * kr_x / B_x / mu_x * (pot_corr_west - pot_corr_east)
        flux_y = res.T_Ry[:,1:-1] * kr_y / B_y / mu_y * (pot_corr_north - pot_corr_south)
        # flux_x = res.T_Rx[1:-1,:] * kr_x / B_x / mu_x * rho_ix * (pot_corr_west - pot_corr_east)
        # flux_y = res.T_Ry[:,1:-1] * kr_y / B_y / mu_y * rho_iy * (pot_corr_north - pot_corr_south)

        flux_x = vstack([zeros([1,flux_x.shape[1]]), flux_x, zeros([1,flux_x.shape[1]])])
        flux_y = hstack([zeros([flux_y.shape[0],1]), flux_y, zeros([flux_y.shape[0],1])])
        return flux_x, flux_y

    @staticmethod
    def residual(res, state, state0, dt):
        flux_x, flux_y = Oil.volume_flux(res, state)
        B_o = Oil.formation_volume_factor(res, state)
        B_o0 = Oil.formation_volume_factor(res, state0)
        div_flux = -(flux_x[1:] - flux_x[:-1] + flux_y[:,1:] - flux_y[:,:-1])
        d_oil = state.Vp  * state.s_o / B_o - state0.Vp * state0.s_o / B_o0
        # d_oil = state.Vp  * state.s_o * Oil.density(res, state) / B_o - state0.Vp * state0.s_o * Oil.density(res, state0)/ B_o0
        # source = zeros(B_o.shape)
        # source[res.well.ix, res.well.iy] = res.well.oil_production(state) 
        source = state.q_o / B_o
        return - d_oil / dt + div_flux + source


class Water:
    @staticmethod
    def density(res, state):
        return res.rho_w_ref * exp(res.c_w * (state.p_w - res.p_ref)) 

    @staticmethod
    def potential(res, state):
        return state.p_w - Water.density(res, state) / 144. * res.z

    @staticmethod
    def viscosity(res, state):
        return res.mu_w * ones(state.p_w.shape)

    @staticmethod
    def formation_volume_factor(res, state):
        B_w = exp(-res.c_w * (state.p_w - res.p_ref))
        # B_w = 1.
        return B_w

    @staticmethod
    def relative_permeability(res, state):
        s_w = state.s_w
        k_w = s_w ** 2
        return k_w

    @staticmethod
    def volume_flux(res, state):
        rho_ix, rho_iy = Reservoir.central_avg(Water.density(res, state))
        pot_corr_east = state.p_w[1:,:] - rho_ix / 144. * res.z[1:,:]
        pot_corr_west = state.p_w[:-1,:] - rho_ix / 144. * res.z[:-1,:]
        pot_corr_south = state.p_w[:,1:] - rho_iy / 144. * res.z[:,1:]
        pot_corr_north = state.p_w[:,:-1] - rho_iy / 144. * res.z[:,:-1]

        kr_x, kr_y = Reservoir.upstream_avg(Water.relative_permeability(res, state),
                                            Water.potential(res, state))
        B_x, B_y = Reservoir.central_avg(Water.formation_volume_factor(res, state))
        # mu_x, mu_y = Reservoir.central_avg(Water.viscosity(res, state))
        # B_x, B_y = 1., 1.
        mu_x, mu_y = res.mu_w, res.mu_w
        # flux_x = res.T_Rx[1:-1,:] * kr_x / B_x / mu_x * rho_ix * (pot_corr_west - pot_corr_east)
        # flux_y = res.T_Ry[:,1:-1] * kr_y / B_y / mu_y * rho_iy * (pot_corr_north - pot_corr_south)
        flux_x = res.T_Rx[1:-1,:] * kr_x / B_x / mu_x * (pot_corr_west - pot_corr_east)
        flux_y = res.T_Ry[:,1:-1] * kr_y / B_y / mu_y * (pot_corr_north - pot_corr_south)

        flux_x = vstack([zeros([1,flux_x.shape[1]]), flux_x, zeros([1,flux_x.shape[1]])])
        flux_y = hstack([zeros([flux_y.shape[0],1]), flux_y, zeros([flux_y.shape[0],1])])
        return flux_x, flux_y

    @staticmethod
    def residual(self, state, state0, dt):
        flux_x, flux_y = Water.volume_flux(res, state)
        B_w = Water.formation_volume_factor(res, state)
        B_w0 = Water.formation_volume_factor(res, state0)

        # source = state.q_w
        source = state.q_w / B_w
        # source = zeros(B_g.shape)
        # source[res.well.ix, res.well.iy] = res.well.gas_production(state) 

        d_water = state.Vp  * state.s_w / B_w - state0.Vp * state0.s_w / B_w0
        # d_water = state.Vp  * state.s_w * Water.density(res, state) / B_w - state0.Vp * state0.s_w * Water.density(res, state0) / B_w0
        div_flux = -(flux_x[1:] - flux_x[:-1] + flux_y[:,1:] - flux_y[:,:-1])
        Res_w = div_flux + source - d_water / dt
        return - d_water / dt + div_flux + source


class Gas:
    @staticmethod
    def density(res, state):
        z_fac = Gas.z_factor(res, state.p_g)
        mwg = res.gas_gravity * 28.96
        return mwg * state.p_g / (res.r_gas * res.temp) / z_fac

    @staticmethod
    def z_factor(res, p_g):
        p_pc = 677 + 15.08 * res.gas_gravity - 37.5 * res.gas_gravity ** 2
        p_pr = p_g / p_pc

        t_pc = 168. + 325 * res.gas_gravity - 12.5 * res.gas_gravity ** 2
        t_pr = (res.temp + 460) / t_pc

        z_fac = 1. - 3.53 * p_pr / (0.983 * t_pr) + 0.274 * p_pr ** 2 / (0.8157 * t_pr)
        return z_fac

    @staticmethod
    def potential(res, state):
        return state.p_g - Gas.density(res, state) / 144. * res.z

    @staticmethod
    def formation_volume_factor(res, state):
        B_g = 14.7 / state.p_g
        return B_g

    @staticmethod
    def relative_permeability(res, state):
        s_g = state.s_g
        k_g = s_g ** 2
        return k_g

    @staticmethod
    def viscosity(res, state):
        return res.mu_g * ones(state.p_o.shape)

    @staticmethod
    def volume_flux(res, state):
        rho_g_ix, rho_g_iy = Reservoir.central_avg(Gas.density(res, state))
        pot_g_corr_east = state.p_g[1:,:] - rho_g_ix / 144. * res.z[1:,:]
        pot_g_corr_west = state.p_g[:-1,:] - rho_g_ix / 144. * res.z[:-1,:]
        pot_g_corr_south = state.p_g[:,1:] - rho_g_iy / 144. * res.z[:,1:]
        pot_g_corr_north = state.p_g[:,:-1] - rho_g_iy / 144. * res.z[:,:-1]

        kr_gx, kr_gy = Reservoir.upstream_avg(Gas.relative_permeability(res, state),
                                              Gas.potential(res, state))
        B_gx, B_gy = Reservoir.central_avg(Gas.formation_volume_factor(res, state))
        R_sx, R_sy = Reservoir.central_avg(state.r_s)
        rho_g_ix, rho_g_iy = Reservoir.central_avg(Gas.density(res, state))
        mu_x, mu_y = Reservoir.central_avg(Gas.viscosity(res, state))

        flux_gx = res.T_Rx[1:-1,:] * kr_gx / B_gx / mu_x * (pot_g_corr_west - pot_g_corr_east)
        flux_gy = res.T_Ry[:,1:-1] * kr_gy / B_gy / mu_y * (pot_g_corr_north - pot_g_corr_south)

        flux_gx = vstack([zeros([1,flux_gx.shape[1]]), flux_gx, zeros([1,flux_gx.shape[1]])])
        flux_gy = hstack([zeros([flux_gy.shape[0],1]), flux_gy, zeros([flux_gy.shape[0],1])])
        return flux_gx, flux_gy

    @staticmethod
    def residual(res, state, state0, dt):
        flux_gx, flux_gy = Gas.volume_flux(res, state)
        flux_ox, flux_oy = Oil.volume_flux(res, state)
        Rs_x, Rs_y = Reservoir.upstream_avg(state.r_s, Gas.potential(res,state))
        flux_ox[1:-1,:] *= Rs_x
        flux_oy[:,1:-1] *= Rs_y
        B_g = Gas.formation_volume_factor(res, state)
        B_o = Oil.formation_volume_factor(res, state)
        B_g0 = Gas.formation_volume_factor(res, state0)
        B_o0 = Oil.formation_volume_factor(res, state0)
        div_g_flux = -(flux_gx[1:] - flux_gx[:-1] + flux_gy[:,1:] - flux_gy[:,:-1])
        div_o_flux = -(flux_ox[1:] - flux_ox[:-1] + flux_oy[:,1:] - flux_oy[:,:-1])

        d_free_gas = state.Vp  * state.s_g / B_g - state0.Vp * state0.s_g / B_g0
        d_solved_gas = state.Vp  * state.s_o / B_o * state.r_s - state0.Vp * state0.s_o / B_o0 * state0.r_s
        source = state.q_g / B_g + state.q_o * state.r_s / B_o
        return -d_free_gas / dt - d_solved_gas / dt + div_g_flux + div_o_flux + source


class ReservoirEquation:
    @staticmethod
    def solve(state0, dt, nNewton, abs_tol, rel_tol, verbose=False):
        # well = state0.res.well
        x = state0.po_sw_sg()
        # R = ReservoirEquation.residual(x, state0,dt)
        # J = R.diff(x)
        po_sw_sg = solve(ReservoirEquation.residual, state0.po_sw_sg(), (state0, dt),max_iter=20, verbose=False)
        state = ReservoirState(po_sw_sg, state0.is_there_gas, res, True)
        state.n_Newton = po_sw_sg._n_Newton
        po_sw_sg.obliviate()
        return state

    @staticmethod
    def residual(po_sw_sg, state0, dt):
        res = state0.res
        state = ReservoirState(po_sw_sg, state0.is_there_gas, res, False)
        Res_o = Oil.residual(res, state, state0, dt)
        Res_w = Water.residual(res, state, state0, dt)
        Res_g = Gas.residual(res, state, state0, dt)
        # r = hstack([ravel([Res_o, Res_w, Res_g]), Res_well])
        return hstack([ravel([Res_o, Res_w, Res_g])])

    # @staticmethod
    # def jacobi(po_sw_sg, state0, dt):
    #     R = ReservoirEquation.residual(po_sw_sg, state0, dt)
    #     J = R.diff(po_sw_sg)
    #     return J

    # def __init__(self, res):
    #     self.res = res
    #     self.waterEqn = Water
    #     self.oilEqn = Oil

    # def solve(self, state0, dt, nNewton, abs_tol, rel_tol, verbose=False):
    #     p_o_s_w_0 = ravel([state0.p_o, state0.s_w])
    #     p_o_s_w = solve(self.residual, p_o_s_w_0, (state0, dt))
    #     self.n_Newton = p_o_s_w._n_Newton
    #     p_o_s_w.obliviate()
    #     return ReservoirState(p_o_s_w, res)

    # def residual(self, p_o_s_w, state0, dt):
    #     state = ReservoirState(p_o_s_w, state0.res)
    #     Res_o = self.oilEqn.residual(res, state, state0, dt)
    #     Res_w = self.waterEqn.residual(res, state, state0, dt)
    #     return ravel([Res_o, Res_w])

class ReservoirHistory:
    def __init__(self, index):
        self.index = index
        self.t = []
        self.q_w = []
        self.q_o = []
        self.q_free_g = []
        self.q_total_g = []
        self.p_o = []
        self.p_wf = []

    def update(self, t, state):
        self.t.append(t)
        self.q_w.append(base((state.q_w/Water.formation_volume_factor(res,state))[self.index]))
        self.q_o.append(base((state.q_o/Oil.formation_volume_factor(res,state))[self.index]))
        self.q_free_g.append(base((state.q_g/Gas.formation_volume_factor(res,state))[self.index]))
        self.q_total_g.append(base((state.q_g/Gas.formation_volume_factor(res,state) + state.q_o/Oil.formation_volume_factor(res,state)*state.r_s)[self.index]))
        self.p_o.append(base(state.p_o[self.index]))
        # q_o = res.well_index * self.lambda_cc_o * (self.p_o - p_wf)
        self.p_wf.append(maximum(base(res.p_bp), base(state.p_o + state.q_o / (res.well_index * state.lambda_cc_o))[self.index]))

    def plot(self):
        subplot(3,1,1)
        # water_lines = plot(self.t, -np.array(self.q_w))
        # oil_lines = plot(self.t, -np.array(self.q_o), '--')
        # gas_lines = plot(self.t, -np.array(self.q_g), '--')
        gas_over_oil = plot(self.t, np.array(self.q_free_g)/np.array(self.q_o))
        # plot(self.t, -np.array(self.q_o)-np.array(self.q_w), ':')
        xlabel('t')
        ylabel('q_g / q_o')
        # legend([water_lines[0], oil_lines[0], gas_lines[0]], ['q_w', 'q_o', 'q_g'])
        subplot(3,1,2)
        water_lines = plot(self.t, -np.array(self.q_w))
        oil_lines = plot(self.t, -np.array(self.q_o), '--')
        gas_lines = plot(self.t, -np.array(self.q_total_g), '--')
        total_lines = plot(self.t, -np.array(self.q_o)-np.array(self.q_w)-np.array(self.q_total_g), ':')
        xlabel('t')
        ylabel('production rate(stc bbl/day)')
        legend([water_lines[0], oil_lines[0], gas_lines[0], total_lines[0]], ['q_w', 'q_o', 'q_g','q_toal'])
        subplot(3,1,3)
        p_gb = plot(self.t, np.array(self.p_o))
        p_wf = plot(self.t, np.array(self.p_wf))
        legend([p_gb[0], p_wf[0]], ['p_o', 'p_wf'])
        draw()


######### Main ###########

# Set initial condition

res = Reservoir(params)

# eqn.verify_Jacobian()
# stop

p = res.p_ref * ones([res.Nx, res.Ny])
s_w_ini1, s_w_ini2 = 0.2, 1.0 - 0.2
s_w = zeros([res.Nx, res.Ny])
# s_w[:2,:] = s_w_ini2
# s_w[2:,:] = s_w_ini1
s_w[:] = s_w_ini1
# s_w[50:,:] = s_w_ini1
s_w[[0,-1],:] = 0.9
s_w[:,[0,-1]] = 0.9

r_s = res.rs_ini * ones([res.Nx, res.Ny])

is_there_gas = np.zeros(p.shape, dtype=bool)

state = ReservoirState(array([p, s_w, r_s]), is_there_gas, res, False)

# Set time 
totalTime = 2000 # day
dt = 0.01 # day
# dt = 1 # day
n_time_steps = int(totalTime / dt)
# n_time_steps = 100 

ds_lim = 0.01
time = 0.
dt_max = 3.
eps = ds_lim / dt_max

hist = ReservoirHistory(res.w_type == 1)
hist.update(time, state)

total_delta_oil, total_delta_water, total_delta_gas = 0., 0., 0.
total_oil, total_water, total_gas = state.total()

# for i_time in range(n_time_steps):
import gc, weakref
gc.collect()
GAR = []
for i_time in range(4):
    state0 = state
    max_Newton_iter, abs_tol, rel_tol = 8, 1E-7, 1E-8
    while True:
        state = ReservoirEquation.solve(state0, dt, max_Newton_iter, abs_tol, rel_tol, verbose=True)

        if state.n_Newton == max_Newton_iter:
            dt /= 2
        else:
            break

    ds_max = (abs(base(state.s_w - state0.s_w))).max(0).max(0) + eps 
    time += dt
    hist.update(time, state)
    total_rate_oil, total_rate_water, total_rate_gas = state.total_rate()
    total_delta_oil += total_rate_oil * dt
    total_delta_water += total_rate_water * dt
    total_delta_gas += total_rate_gas * dt
    total_oil0, total_water0, total_gas0 = total_oil, total_water, total_gas
    total_oil, total_water, total_gas = state.total()
    # print(i_time, time, dt, -total_delta_oil+total_oil, -total_delta_water+total_water, -total_delta_gas+total_gas)
    # print(i_time, time, dt, -total_delta_oil+total_oil, -total_delta_water+total_water, -total_delta_gas+total_gas, state.p_o[res.w_type == 1],state.p_w[res.w_type == 1])

    # if time > 2:
    #     res.w_type[int(res.Nx*3/10),int(res.Ny*7/10)] = 1
    #     res.w_type[int(res.Nx*7/10),int(res.Ny*3/10)] = 1

    # adjust time step size
    if ds_max > ds_lim:
        dt = dt * ds_lim / ds_max
    elif ds_max < 0.5 * ds_lim and state.n_Newton <= 0.8 * max_Newton_iter:
        dt = dt * min(1.5, 0.5 * ds_lim / ds_max)

    # if i_time % 10 == 0:
    #     x = tile(base(res.x_cc[:,newaxis]), (1, res.Ny))
    #     y = tile(base(res.y_cc[newaxis,:]), (res.Nx, 1))
    #     u_w, v_w = state.water_velocity()
    #     u_o, v_o = state.oil_velocity()

    #     figure(1, figsize=(10, 10))
    #     # figure(1, figsize=(10, 10))
    #     clf()
    #     suptitle('days: {0}'.format(time))
    #     subplot(2,2,1)
    #     title('s_w')
    #     contourf(x.T, y.T, base(state.s_w.T), 100)
    #     colorbar()
    #     # streamplot(x.T, y.T, u_w.T, v_w.T, density=2)
    #     subplot(2,2,2)
    #     title('s_o')
    #     contourf(x.T, y.T, base(state.s_o.T), 100)
    #     colorbar()
    #     # streamplot(x.T, y.T, u_o.T, v_o.T, density=2)
    #     subplot(2,2,3)
    #     title('s_g')
    #     contourf(x.T, y.T, base(state.s_g.T), 100)
    #     colorbar()
    #     subplot(2,2,4)
    #     title('p_o')
    #     contourf(x.T, y.T, base(state.p_o.T), 100)
    #     colorbar()
    #     # streamplot(x.T, y.T, u_w.T, v_w.T, density=2)
    #     # subplot(2,2,4)
    #     # title('p_o')
    #     # contourf(x.T, y.T, base(state.p_o.T), 100)
    #     # colorbar()
    #     # streamplot(x.T, y.T, base(u_o).T, base(v_o).T, density=2)
    #     draw()
    #     savefig('fig/streamline{0:06d}'.format(i_time))

    #     figure(2, figsize=(8, 8))
    #     clf()
    #     hist.plot()
    #     savefig('fig/pro_wells1{0:06d}'.format(i_time))

    #     def facecolors(s_w):
    #         s = (s_w[1:,1:] + s_w[:-1,1:] + s_w[1:,:-1] + s_w[:-1,:-1]) / 4
    #         jet = get_cmap('jet')
    #         # colors = empty(s.shape, dtype=str)
    #         # colors[s > 0.5] = 'b'
    #         # colors[s <= 0.5] = 'r'
    #         colors = jet(1-s)
    #         return colors

    #     # fig = figure(3, figsize=(8, 8))
    #     # clf()
    #     # ax = fig.gca(projection='3d')
    #     # ax.plot_surface(x, y, res.z, rstride=1, cstride=1,
    #     #                 facecolors=facecolors(state.s_w))
    #     # ax.set_aspect('equal','box')
    #     # draw()
    #     # savefig('fig/surface{0:06d}'.format(i_time))
    #     # sleep(0.5)

    if time > totalTime:
        break
    # print(i_time*dt)
    # print(i_time*dt, p_o[0], p_o[-1], s_w[0], s_w[-1], linalg.norm(Res_o), linalg.norm(Res_w))

    print('garbage collected = ', gc.collect())
    GAR.append([weakref.ref(obj) for obj in gc.get_objects()
                if isinstance(obj, adarray)])
    print('total objects = ', len(GAR[-1]))


def in_first_but_not_second(weak_list1, weak_list2):
    list_1_over_2 = []
    for a in weak_list1:
        if a() is None:
            break
        found_in_2 = False
        for b in weak_list2:
            if a() is b():
                found_in_2 = True
                break
        if not found_in_2:
            list_1_over_2.append(a)
    return list_1_over_2

G = [g() for g in GAR[0] if g() is not None]
