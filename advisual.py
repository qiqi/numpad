import unittest
import networkx as nx
from adarray import *
from adsolve import *

def _build_graph_recurse(G, f, name_dict, i_f_ops):

    def find_name(f):
        for key in name_dict:
            if name_dict[key][0] is f:
                return key
        # create new name
        key = 'a{0}:{1}'.format(len(name_dict), str(f.shape))
        name_dict[key] = (f, set())
        return key

    f_name = find_name(f)
    node_name = f_name + ':{0}'.format(i_f_ops)

    f_ops = name_dict[f_name][1]
    if not i_f_ops in f_ops:
        f_ops.add(i_f_ops)
        G.add_node(node_name)
        # build subtree
        if i_f_ops > 0:
            op = f._ops[i_f_ops - 1]
            if len(op) == 1:  # self operation
                multiplier = op[0]
                if multiplier is not 0:
                    other_name = _build_graph_recurse(G, f, name_dict,
                                                      i_f_ops - 1)
                    G.add_edge(node_name, other_name)
            else:
                other, i_other_ops, multiplier = op
                other_name = _build_graph_recurse(G, other, name_dict,
                                                  i_other_ops)
                G.add_edge(node_name, other_name)
                other_name = _build_graph_recurse(G, f, name_dict,
                                                  i_f_ops - 1)
                G.add_edge(node_name, other_name)
        # f is an adsolution
        elif hasattr(f, '_residual'):
            other_name = _build_graph_recurse(G, f._residual, name_dict,
                                                 f._residual_ops)
            G.add_edge(node_name, other_name)
    return node_name

def build_graph(f):
    G = nx.DiGraph()
    _build_graph_recurse(G, f, {}, f.i_ops())
    return G

def draw_spectral(f):
    G = build_graph(f)
    nx.draw_spectral(G, font_size=20)



if __name__ == '__main__':
    # visualize 5 steps of Euler flow calculation

    def extend(w_interior):
        w = zeros([4, Ni+2, Nj+2])
        w[:,1:-1,1:-1] = w_interior.reshape([4, Ni, Nj])
        # inlet
        rho, u, v, E, p = primative(w[:,1,1:-1])
        c = sqrt(1.4 * p / rho)
        mach = u / c
        rhot = rho * (1 + 0.2 * mach**2)**2.5
        pt = p * (1 + 0.2 * mach**2)**3.5
    
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
        w[:3,-1,1:-1] = w[:3,-2,1:-1]
        w[3,-1,1:-1] = p_out / (1.4 - 1) + \
                    0.5 * (w[1,-1,1:-1]**2 + w[2,-1,1:-1]**2) / w[0,-1,1:-1]
        # walls
        w[:,:,0] = w[:,:,1]
        w[2,:,0] *= -1
        w[:,:,-1] = w[:,:,-2]
        w[2,:,-1] *= -1
        return w
        
    def primative(w):
        rho = w[0]
        u = w[1] / rho
        v = w[2] / rho
        E = w[3]
        p = 0.4 * (E - 0.5 * (u * w[1] + v * w[2]))
        return rho, u, v, E, p
        
    def euler(w, w0, dt):
        w_ext = extend(w)
        rho, u, v, E, p = primative(w_ext)
        # cell center flux
        F = array([rho*u, rho*u**2 + p, rho*u*v, u*(E + p)])
        G = array([rho*u, rho*u*v, rho*v**2 + p, v*(E + p)])
        # interface flux
        Fx = 0.5 * (F[:,1:,1:-1] + F[:,:-1,1:-1])
        Fy = 0.5 * (F[:,1:-1,1:] + F[:,1:-1,:-1])
        # numerical viscosity
        C = 300
        Fx -= 0.5 * C * (w_ext[:,1:,1:-1] - w_ext[:,:-1,1:-1])
        Fy -= 0.5 * C * (w_ext[:,1:-1,1:] - w_ext[:,1:-1,:-1])
        # residual
        divF = (Fx[:,1:,:] - Fx[:,:-1,:]) / dx + (Fy[:,:,1:] - Fy[:,:,:-1]) / dy
        return (w - w0) / dt + ravel(divF)

    # ---------------------- time integration --------------------- #
    Ni, Nj = 10, 5
    dx, dy = 10./ Ni, 1./Nj
    t, dt = 0, 1E-5
    
    pt_in = 1.2E5
    p_out = 1E5
    
    w = zeros([4, Ni, Nj])
    w[0] = 1
    w[3] = 1E5 / (1.4 - 1)
    
    w = ravel(w)
    
    for i in range(5):
        print('t = ', t)
        w0 = w
        w = solve(euler, w0, args=(w0, dt), rel_tol=1E-9, abs_tol=1E-7)
        if w._n_Newton == 1:
            break
        elif w._n_Newton < 4:
            w0 = w
            dt *= 2
        elif w._n_Newton < 8:
            w0 = w
        else:
            dt *= 0.5
            continue
        t += dt

    import pylab
    draw_spectral(w)
    pylab.show()
