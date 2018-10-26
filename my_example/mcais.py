# external imports
import numpy as np
import matplotlib.pyplot as plt

# internal imports
from pympc.geometry.polyhedron import Polyhedron
from pympc.dynamics.discrete_time_systems import LinearSystem
from pympc.plot import plot_state_space_trajectory

def main():
    # ===============
    # Model Parameter
    # ===============
    m = 1
    l = 1
    g = 10.
    # ============
    # Inv Pend Dyn
    # ============
    A = np.array([
        [ 0., 1.],
        [g/l, 0.]
        ])
    B = np.array([
        [          0.],
        [1./(m*l**2.)]
        ])
    h = 0.01
    S = LinearSystem.from_continuous(A, B, h, 'zero_order_hold')

    # =========
    # Solve LQR
    # =========
    Q = np.array([[1., 0.],[0., 1.]])
    R = np.array([[1.]])
    P, K = S.solve_dare(Q, R)

    # ===============
    # Polytopic Const
    # ===============
    x_min = np.array([-1., -1.])
    x_max = np.array([ 1., 1.])
    X = Polyhedron.from_bounds(x_min, x_max)
    u_min = np.array([-15.])
    u_max = np.array([15.])
    U = Polyhedron.from_bounds(u_min, u_max)

    # =====
    # MCAIS
    # =====
    D = X.cartesian_product(U)
    O_inf = S.mcais(K, D)

    # ====
    # X_cl
    # ====

    U_cl = Polyhedron(U.A.dot(K), U.b)
    X_cl = X.intersection(U_cl)

    # =============
    # Plot polytope
    # =============

    X.plot(label=r'$\mathcal{X}$', facecolor='b')
    X_cl.plot(label=r'$\mathcal{X}_{\mathrm{cl}}$', facecolor='g')
    O_inf.plot(label=r'$\mathcal{O}_{\infty}$', facecolor='r')

    # ================
    # plot simulations
    # ================

    N = 100
    n_sim = 50
    case = 5
    save = True
    if case == 1:
        x_1 = .5
        for x_0 in np.linspace(x_min[0], x_max[0], n_sim):
            x = np.array([x_0, x_1])
            x_trajectory = S.simulate_closed_loop(x, N, K)
            plot_state_space_trajectory(x_trajectory, color='c')
    elif case == 2:
        x_1 = -0.5
        for x_0 in np.linspace(x_min[0], x_max[0], n_sim):
            x = np.array([x_0, x_1])
            x_trajectory = S.simulate_closed_loop(x, N, K)
            plot_state_space_trajectory(x_trajectory, color='c')
    elif case == 3:
        x_0 = 0.77
        for x_1 in np.linspace(x_min[1], x_max[1], n_sim):
            x = np.array([x_0, x_1])
            x_trajectory = S.simulate_closed_loop(x, N, K)
            plot_state_space_trajectory(x_trajectory, color='c')
    elif case == 4:
        x_0 = -0.77
        for x_1 in np.linspace(x_min[1], x_max[1], n_sim):
            x = np.array([x_0, x_1])
            x_trajectory = S.simulate_closed_loop(x, N, K)
            plot_state_space_trajectory(x_trajectory, color='c')
    elif case == 5:
        for i in range(n_sim):
            x = np.multiply(np.random.rand(2), x_max - x_min) + x_min
            x_trajectory = S.simulate_closed_loop(x, N, K)
            plot_state_space_trajectory(x_trajectory, color='c')
    plt.legend()
    if save:
        plt.savefig('figures/mcais/mcais_case_'+str(case)+'.pdf', bbox_inches='tight')
        print("Figure saved")
    else:
        plt.show()

if __name__ == "__main__":
    main()
