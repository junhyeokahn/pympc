import numpy as np
import scipy.linalg as linalg
from itertools import product
from copy import copy
from ad import adnumber, jacobian
from collections import OrderedDict
from pympc.geometry.polytope import Polytope
from pympc.dynamical_systems import DTAffineSystem, DTPWASystem
from pympc.control import MPCHybridController
from pympc.optimization.pnnls import linear_program
from pympc.models.boxatlas_visualizer import BoxAtlasVisualizer
import boxatlas_parameters

class MovingLimb():
    def __init__(self, A_domains, b_domains, contact_surfaces, nominal_position, forbidden_transitions=[]):

        # copy inputs
        self.modes = A_domains.keys()
        self.n_domains = len(A_domains)
        self.A_domains = {}
        self.b_domains = {}
        self.contact_surfaces = contact_surfaces
        self.nominal_position = nominal_position
        self.forbidden_transitions = forbidden_transitions

        # normalize all the A and all the b
        for mode in A_domains.keys():
            norm_factor = np.linalg.norm(A_domains[mode], axis=1)
            self.A_domains[mode] = np.divide(A_domains[mode].T, norm_factor).T
            self.b_domains[mode] = np.divide(b_domains[mode].T, norm_factor).T

        return

class FixedLimb():
    def __init__(self, position, normal, nominal_force):
        self.position = position
        self.normal = normal/np.linalg.norm(normal)
        self. nominal_force = nominal_force
        return

class Trajectory:
    
    def __init__(self, x, u, Q, R, P):
        self.x = x
        self.u = u
        self.Q = Q
        self.R = R
        self.P = P
        self.cost = self._cost()
        return
        
    def _cost(self):
        cost = 0
        for u in self.u:
            cost += u.T.dot(self.R).dot(u)
        for x in self.x[:-1]:
            cost += x.T.dot(self.Q).dot(x)
        cost += self.x[-1].T.dot(self.P).dot(self.x[-1])
        return cost

class BoxAtlas():

    def __init__(self, limbs, nominal_mode):
        self.limbs = limbs
        self.nominal_mode = nominal_mode
        self.x_eq = self._get_equilibrium_state()
        self.u_eq = self._get_equilibrium_input()
        self.n_x = self.x_eq.shape[0]
        self.n_u = self.u_eq.shape[0]
        self.x_diff = self._differentiable_state()
        self.u_diff = self._differentiable_input()
        self.contact_modes = self._get_contact_modes()
        domains = self._get_domains()
        self.contact_modes, domains = self._remove_empty_domains(domains)
        affine_systems = self._get_affine_systems()
        self.pwa_system = DTPWASystem(affine_systems, domains)
        self.nominal_system, self.nominal_domain = self._extract_nominal_configuration()
        self._check_equilibrium_point()
        self._visualizer = self._initialize_visualizer()
        return

    def _get_equilibrium_state(self):
        """
        Translates the equilibrium_configuration dictionary in the equilibrium state vector x_equilibrium.
        """

        # moving limbs position
        x_equilibrium = np.zeros((0,1))
        for limb in self.limbs['moving'].values():
            x_equilibrium = np.vstack((x_equilibrium, limb.nominal_position))

        # body state (limbs + body position and velocity)
        x_equilibrium = np.vstack((x_equilibrium, np.zeros((4,1))))

        return x_equilibrium

    def _get_equilibrium_input(self):
        """
        Translates the equilibrium_limb_forces dictionary in the equilibrium input vector u_equilibrium.
        """

        # velocities of the moving limbs
        u_equilibrium = np.zeros((len(self.limbs['moving'])*2, 1))

        # forces of the fixed limbs
        for limb in self.limbs['fixed'].values():
            u_equilibrium = np.vstack((u_equilibrium, limb.nominal_force))

        return u_equilibrium

    def _get_contact_modes(self):
        """
        Returns a list of dictionaries. Each dictionary represents a mode of box-atlas: the keys of the dict are the names of the moving limbs, the values of the dict are the names given to the modes of the limbs, e.g.:
        contact_modes = [
        {'left_hand': 'not_in_contact', 'right_hand': 'not_in_contact'},
        {'left_hand': 'in_contact', 'right_hand': 'not_in_contact'},
        {'left_hand': 'not_in_contact_side', 'right_hand': 'in_contact'}
        ]
        """

        # list of modes of each moving limb
        moving_limbs_modes = [self.limbs['moving'][limb].modes for limb in self.limbs['moving'].keys()]

        # list of modes of atlas (tuples)
        modes_tuples = product(*moving_limbs_modes)

        # reorganization of atlas's modes into a list of dicts
        contact_modes = []
        for mode_tuple in modes_tuples:
            mode = {}
            for i, limb_mode in enumerate(mode_tuple):
                mode[self.limbs['moving'].keys()[i]] = limb_mode
            contact_modes.append(mode)

        return contact_modes

    def _differentiable_state(self):
        """
        Defines a dictionary with the elements of the state using differentiable arrays from the AD package.
        """

        x_diff = OrderedDict()

        # moving limbs
        for limb in self.limbs['moving'].keys():
            x_diff['q'+limb] = adnumber(np.zeros((2,1)))

        # body
        x_diff['qb'] = adnumber(np.zeros((2,1)))
        x_diff['vb'] = adnumber(np.zeros((2,1)))

        return x_diff

    def _differentiable_input(self):
        """
        Defines a dictionary with the elements of the input using differentiable arrays from the AD package.
        """

        u_diff = OrderedDict()

        # moving limbs
        for limb in self.limbs['moving'].keys():
            u_diff['v'+limb] = adnumber(np.zeros((2,1)))

        # fixed limbs
        for limb in self.limbs['fixed'].keys():
            u_diff['f'+limb] = adnumber(np.zeros((2,1)))

        return u_diff

    def _get_domains(self):
        """
        Reutrn the list of the domains in the (x,u) space for each mode of atlas. Each domain is a polytope and is derived differentiating the linear constraints. The list 'constraints' contains a set of vector inequalities in the variables (x_diff, u_diff); each inequality has to be interpreted as:
        f(x_diff, u_diff) <= 0
        and being each f(.) a linear function, we have equivalently
        grad(f) * (x_diff, u_diff) <= - f(0,0)
        """

        # mode independent constraints
        constraints = []
        constraints = self._state_constraints(constraints)
        constraints = self._input_constraints(constraints)

        # mode dependent constraints
        domains = []
        for mode in self.contact_modes:
            mode_constraint = self._contact_mode_constraints(mode, copy(constraints))
            A, B, c = self._matrices_from_linear_expression(mode_constraint)
            lhs = np.hstack((A, B))
            rhs = - c
            D = Polytope(lhs, rhs - lhs.dot(np.vstack((self.x_eq, self.u_eq))))
            D.assemble()
            domains.append(D)

        return domains

    def _state_constraints(self, constraints):
        """
        Generates differentiable expressions for the constraints on the state that don't depend on the mode.
        """

        # moving limbs
        for limb in self.limbs['moving'].keys():
            constraints.append(self.x_diff['q'+limb] - self.x_diff['qb'] - boxatlas_parameters.joint_limits[limb]['max'])
            constraints.append(boxatlas_parameters.joint_limits[limb]['min'] - self.x_diff['q'+limb] + self.x_diff['qb'])

        # fixed limbs
        for limb_key, limb_value in self.limbs['fixed'].items():
            constraints.append(limb_value.position - self.x_diff['qb'] - boxatlas_parameters.joint_limits[limb_key]['max'])
            constraints.append(boxatlas_parameters.joint_limits[limb_key]['min'] - limb_value.position + self.x_diff['qb'])

        # body
        constraints.append(self.x_diff['qb'] - boxatlas_parameters.joint_limits['b']['max'])
        constraints.append(boxatlas_parameters.joint_limits['b']['min'] - self.x_diff['qb'])
        constraints.append(self.x_diff['vb'] - boxatlas_parameters.velocity_limits['b']['max'])
        constraints.append(boxatlas_parameters.velocity_limits['b']['min'] - self.x_diff['vb'])

        return constraints

    def _input_constraints(self, constraints):
        """
        Generates differentiable expressions for the constraints on the input that don't depend on the mode.
        """

        # moving limbs
        for limb in self.limbs['moving'].keys():
            constraints.append(self.u_diff['v'+limb] - boxatlas_parameters.velocity_limits[limb]['max'])
            constraints.append(boxatlas_parameters.velocity_limits[limb]['min'] - self.u_diff['v'+limb])

        # fixed limbs
        mu  = boxatlas_parameters.friction
        for limb in self.limbs['fixed'].keys():
            constraints.append(self.u_diff['f'+limb] - boxatlas_parameters.force_limits[limb]['max'])
            constraints.append(boxatlas_parameters.force_limits[limb]['min'] - self.u_diff['f'+limb])

            # friction limits
            constraints.append(-mu*self.u_diff['f'+limb][0,0] + self.u_diff['f'+limb][1,0])
            constraints.append(-mu*self.u_diff['f'+limb][0,0] - self.u_diff['f'+limb][1,0])

        return constraints

    def _contact_mode_constraints(self, mode, constraints):
        """
        Adds to the list of contraints the constraints that are mode dependent.
        """
        
        mu  = boxatlas_parameters.friction
        for limb in self.limbs['moving'].keys():

            # moving limbs in the specified domain
            A = self.limbs['moving'][limb].A_domains[mode[limb]]
            b = self.limbs['moving'][limb].b_domains[mode[limb]]
            constraints.append(A.dot(self.x_diff['q'+limb]) - b)

            # friction constraints (surface equation: n^T x <= d)
            contact_surface = self.limbs['moving'][limb].contact_surfaces[mode[limb]]
            if contact_surface is not None:
                n = A[contact_surface, :].reshape(2,1) # normal vector
                t = np.array([[-n[1,0]],[n[0,0]]]) # tangential vector
                d = b[contact_surface, 0].reshape(1,1) # offset
                f_n = boxatlas_parameters.stiffness*(d - n.T.dot(self.x_diff['q'+limb])) # normal force (>0)
                f_t = - boxatlas_parameters.damping*(t.T.dot(self.u_diff['v'+limb])) # tangential force
                constraints.append(f_t - mu*f_n)
                constraints.append(- f_t - mu*f_n)

        return constraints

    def _remove_empty_domains(self, domains):
        """
        Removes from the domains and the contact modes all the modes that have an empty domain.
        """
        non_empty_contact_modes = []
        non_empty_domains = []
        for i, D in enumerate(domains):
            if not D.empty:
                non_empty_contact_modes.append(self.contact_modes[i])
                non_empty_domains.append(D)
        return non_empty_contact_modes, non_empty_domains

    def _get_affine_systems(self):
        """
        Returns the list of affine systems, one for each mode of the robot.
        """
        affine_systems = []
        for mode in self.contact_modes:
            dynamics = self._continuous_time_dynamics(mode)
            A_ct, B_ct, c_ct = self._matrices_from_linear_expression(dynamics)
            sys = DTAffineSystem.from_continuous(
                A_ct,
                B_ct,
                c_ct + A_ct.dot(self.x_eq) + B_ct.dot(self.u_eq), # traslates the equilibrium to the origin
                boxatlas_parameters.sampling_time,
                boxatlas_parameters.integrator
                )
            affine_systems.append(sys)
        return affine_systems

    def _continuous_time_dynamics(self, mode):
        """
        Returns the right hand side of the dynamics
        \dot x = f(x, u)
        where x and u are AD variables and f(.) is a linear function.
        """

        # position of the moving limbs (controlled in velocity)
        dynamics = []
        for limb in self.limbs['moving'].keys():
            dynamics.append(self.u_diff['v'+limb])

        # position of the body (simple integrator)
        dynamics.append(self.x_diff['vb'])

        # velocity of the body
        g = np.array([[0.],[-boxatlas_parameters.gravity]])
        v_dot = adnumber(g)
        for limb in self.limbs['moving'].keys():
            v_dot = v_dot + self._force_moving_limb(limb, mode)/boxatlas_parameters.mass
        for limb in self.limbs['fixed'].keys():
            v_dot = v_dot + self._force_fixed_limb(limb)/boxatlas_parameters.mass
        dynamics.append(v_dot)

        return dynamics

    def _force_moving_limb(self, limb, mode):
        """
        For a given moving limb, in a given mode, returns the contact force f(x, u) (= f_n + f_t), where x and u are AD variables and f(.) is a linear function.
        """

        # domain
        A = self.limbs['moving'][limb].A_domains[mode[limb]]
        b = self.limbs['moving'][limb].b_domains[mode[limb]]
        contact_surface = self.limbs['moving'][limb].contact_surfaces[mode[limb]]

        # free space
        if contact_surface is None:
            f = adnumber(np.zeros((2,1)))

        # elastic wall
        else:
            n = A[contact_surface, :].reshape(2,1) # normal vector
            d = b[contact_surface, 0].reshape(1,1) # offset
            penetration = n.T.dot(self.x_diff['q'+limb]) - d
            f_n = - boxatlas_parameters.stiffness * penetration * n
            v_t = self.u_diff['v'+limb] - (self.u_diff['v'+limb].T.dot(n)) * n
            f_t = - boxatlas_parameters.damping * v_t
            f = f_n + f_t

        return f

    def _force_fixed_limb(self, limb):
        """
        For a given fixed limb, returns the contact force f(x, u) (= f_n + f_t), where x and u are AD variables and f(.) is a linear function.
        """
        n = self.limbs['fixed'][limb].normal
        t = np.array([[-n[1,0]],[n[0,0]]])
        f_n = n * self.u_diff['f'+limb][0,0]
        f_t = t * self.u_diff['f'+limb][1,0]
        f = f_n + f_t
        return f

    def _matrices_from_linear_expression(self, f):
        """
        Takes a linear expression in the form
        f(x, u)
        and returns (A, B, c) such that
        f(x, u) = A x + B u + c
        """

        # variables
        x = np.vstack(*[self.x_diff.values()])
        u = np.vstack(*[self.u_diff.values()])

        # concatenate if list
        if type(f) is list:
            f = np.vstack(f)

        # get matrices
        A = np.array(jacobian(f.flatten().tolist(), x.flatten().tolist()))
        B = np.array(jacobian(f.flatten().tolist(), u.flatten().tolist()))
        c = np.array([[el[0].x] for el in f])

        return A, B, c

    def _extract_nominal_configuration(self):
        for i, mode in enumerate(self.contact_modes):
            if mode == self.nominal_mode:
                nominal_mode_index = i
        nominal_system = self.pwa_system.affine_systems[nominal_mode_index]
        nominal_domain = self.pwa_system.domains[nominal_mode_index]
        return nominal_system, nominal_domain


    def _check_equilibrium_point(self, tol= 1.e-6):
        """
        Check that the nominal configuration of the moving limbs and the nominal forces of the fixed limbs are an equilibrium point for atlas in the nominal mode.
        """
        offset_norm = np.linalg.norm(self.nominal_system.c)
        if offset_norm > tol:
            raise ValueError('Given nominal configuration and forces are not an equilibrium state.')
        return

    def penalize_relative_positions(self, Q_rel):
        """
        Given the matrix Q_rel which is intended to minimize
        x_rel' Q_rel x_rel
        where x_rel = (q_moving_limb - q_body, q_body, v_body), returns the matrix Q such that
        x_rel' Q_rel x_rel = x' Q x
        with x = (q_moving_limb, q_body, v_body).
        """
        x_rel = [self.x_diff['q'+limb] - self.x_diff['qb'] for limb in self.limbs['moving'].keys()]+[self.x_diff['qb']]+[self.x_diff['vb']]
        x_rel = np.vstack(x_rel)
        x = np.vstack(*[self.x_diff.values()])
        cost = x_rel.T.dot(Q_rel).dot(x_rel)[0,0]
        Q = .5*np.array(cost.hessian(x.flatten().tolist()))
        return Q

    def is_inside_a_domain(self, x):
        """
        Checks if there exists an input vector u such that the given state is inside of at least one domain.
        """
        is_inside = False
        for D in self.pwa_system.domains:
            A_x = D.lhs_min[:,:self.n_x]
            A_u = D.lhs_min[:,self.n_x:]
            b_u = D.rhs_min - A_x.dot(x)
            cost = np.zeros((self.n_u, 1))
            sol = linear_program(cost, A_u, b_u)
            if not np.isnan(sol.min):
                is_inside = True
                break
        return is_inside

    def _initialize_visualizer(self):
        walls = []
        for limb in self.limbs['moving'].values():
            for mode in limb.modes:
                if limb.contact_surfaces[mode] is not None:
                    A = limb.A_domains[mode]
                    b = limb.b_domains[mode]
                    wall = Polytope(A, b)
                    wall.add_bounds(boxatlas_parameters.visualizer_min, boxatlas_parameters.visualizer_max)
                    wall.assemble()
                    walls.append(wall)
        for limb in self.limbs['fixed'].values():
            support_box = self._box_from_surface(limb.position, limb.normal, limb.normal.T.dot(limb.position))
            walls.append(support_box)
        visualizer = BoxAtlasVisualizer(walls, self.limbs)
        return visualizer

    @staticmethod
    def _box_from_surface(x, n, d, depth=.05, width=.1):
        t = np.array([[-n[1,0]],[n[0,0]]])
        c = t.T.dot(x)
        lhs = np.vstack((n.T, -n.T, t.T, -t.T))
        rhs = np.vstack((d, -d+depth, c+width, -c+width))
        box = Polytope(lhs, rhs)
        box.assemble()
        return box

    def visualize(self, x):
        configuration = self._configuration_to_visualizer(x)
        self._visualizer.visualize(configuration)
        return

    def _configuration_to_visualizer(self, x):
        configuration = dict()
        for i, limb in enumerate(self.limbs['moving'].keys()):
            configuration[limb] = x[i*2:(i+1)*2, :]
        configuration['b'] = x[(i+1)*2:(i+2)*2, :]
        return configuration

    def state_labels(self):
        x = []
        for limb in self.limbs['moving'].keys():
            x += ['q' + limb + 'x', 'q' + limb + 'y']
        x += ['qbx', 'qby', 'vbx', 'vby']
        print 'Box-Atlas states:\n', x
        return

    def input_labels(self):
        u = []
        for limb in self.limbs['moving'].keys():
            u += ['v' + limb + 'x', 'v' + limb + 'y']
        for limb in self.limbs['fixed'].keys():
            u += ['f' + limb + 'n', 'f' + limb + 't']
        print 'Box-Atlas inputs:\n', u
        return

    def avoid_forbidden_transitions(self, controller):
        for limb_key, limb_value in self.limbs['moving'].items():
            for transition in limb_value.forbidden_transitions:
                previous_modes = [i for i, mode in enumerate(self.contact_modes) if mode[limb_key] == transition[0]]
                next_modes = [i for i, mode in enumerate(self.contact_modes) if mode[limb_key] == transition[1]]
                for previous_mode in previous_modes:
                    for next_mode in next_modes:
                        for k in range(controller.N-1):
                            expr = controller._d[k, previous_mode] + controller._d[k+1, next_mode]
                            controller._model.addConstr(expr <= 1.)
        return controller