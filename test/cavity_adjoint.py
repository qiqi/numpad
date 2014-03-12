from cavity import *

print('Solving the adjoint')

# adjoint of kinetic energy
kinetic_energy = 0.5 * sum(u_and_p[N*N:]**2)

adj = np.array(kinetic_energy.diff(force)).ravel()
adj_x = adj[N*N:N*(2*N-1)].reshape([N-1,N])
adj_y = adj[N*(2*N-1):].reshape([N,N-1])

adj_x = 0.5 * vstack([adj_x[:1,:], adj_x[1:,:] + adj_x[:-1,:], adj_x[-1:,:]])
adj_y = 0.5 * hstack([adj_y[:,:1], adj_y[:,1:] + adj_y[:,:-1], adj_y[:,-1:]])

figure()
streamplot(x, y, adj_x.T, adj_y.T, density=5)
title('adjoint')
axis('scaled')
axis([0,1,0,1])
xlabel('x')
ylabel('y')

savefig('cavity{0:d}_adjoint.png'.format(Re))
