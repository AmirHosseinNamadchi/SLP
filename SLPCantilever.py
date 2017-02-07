#
# Author: Amir Hossein Namadchi, February 2017
#
"""
Solving a simple nonlinear optimization problem using sequential linear
programming (SLP) method.
"""

import numpy as np
from scipy.optimize import linprog

class Problem:
    """Defines the problem specifications

	    Keyword arguments:
	    rho -- the density (default 1)
	    L -- length of beam (default 2000)
	    P -- point load value  (default 20000)
	    E -- young's modulus (default 21e4)
    """
    def __init__(self, rho = 1, L = 2000, P = 20000, E = 21e4):
        self.rho = rho
        self.L = L
        self.P = P
        self.E = E
    
    @staticmethod
    def A(w, t):
        """Cross-sectional area

	        Keyword arguments:
	        w -- section height
	        t -- thickness
        """
        return 4*t*(-t+w)

    @staticmethod
    def Q(w, t):
        """1st moment of inertia

	        Keyword arguments:
	        w -- section height
	        t -- thickness
        """
        return (t*(4*t**2-6*t*w+3*w**2))/4

    @staticmethod
    def I(w, t):
        """2nd moment of inertia

	        Keyword arguments:
	        w -- section height
	        t -- thickness
        """
        return (w**4-(-2*t+w)**4)/12


class SLPOptimization:
    """SLP & optimization parameters (design variables are w and t)
        
        Keyword arguments:
        delta_max -- maximum allowed tip displacement (default 10)
        sigma_max -- maximum allowed bending stress (default 165)
        tau_max -- maximum allowed shear stress (default 90)
        lb -- lower bounds of design variables (default [60,10])
        ub -- upper bounds of design variables (default [300,40])
        AL -- coefficient matrix of linear constraints (default [1,-8])
        bl -- corresponding constant terms (AL.x = bl) (default [0])
        tol -- a prescribed small positive tolerance (defaul 0.0001)
        max_iter -- maximum number of iterations (default 20)
    """

    def __init__(self, delta_max = 10, sigma_max = 165, tau_max = 90, \
                lb = np.array([60,10]), ub = np.array([300,40]), \
                AL = np.array([1,-8]), bL = np.array([0]), tol = 0.0001, \
                max_iter = 20):
        self.delta_max = delta_max
        self.sigma_max = sigma_max
        self.tau_max = tau_max
        self.lb = lb
        self.ub = ub
        self.AL = AL
        self.bL = bL
        self.tol = tol
        self.max_iter = max_iter
    
    @staticmethod
    def cost_function(problem_object, w, t):
        """Cost function (weight)

	        Keyword arguments:
            problem_object -- an instance of Problem class
	        w -- section height
	        t -- thickness
        """
        return problem_object.rho*problem_object.A(w,t)*problem_object.L

    def displacement_cons(self, problem_object, w, t):
        """Displacement constraint 
        
            Keyword arguments:
            problem_object -- an instance of Problem class
	        w -- section height
	        t -- thickness
        """
        g_d = (problem_object.P*(problem_object.L**3))/(3*problem_object.E\
            *problem_object.I(w,t)) - self.delta_max
        return g_d

    def sigma_cons(self, problem_object, w, t):
        """Bending stress constraint 
        
            Keyword arguments:
            problem_object -- an instance of Problem class
	        w -- section height
	        t -- thickness
        """
        g_b = (problem_object.P*problem_object.L*w)\
             /(2*problem_object.I(w,t)) - self.sigma_max
        return g_b

    def tau_cons(self, problem_object, w, t):
        """Shear stress constraint 
        
            Keyword arguments:
            problem_object -- an instance of Problem class
	        w -- section height
	        t -- thickness
        """
        g_s = (problem_object.P*problem_object.Q(w,t))\
             /(2*t*problem_object.I(w,t)) - self.tau_max
        return g_s

    @staticmethod
    def calculate_grad(problem_object, function, w, t):
        """Calculates approximate gradient of function
        """
        eps = 0.00001
        dfdw = (function(problem_object, w + eps, t) - 
                function(problem_object, w, t))/eps
        dfdt = (function(problem_object, w, t + eps) - 
                function(problem_object, w, t))/eps
        return dfdw, dfdt
    
    def run_SLP(self, problem_object, w, t):
        """Runs SLP algorithm with initial guess w and t 
        """
        x = np.array([w,t])
        grad_gd = np.asarray(self.calculate_grad(problem_object, self.displacement_cons, w, t))
        grad_gb = np.asarray(self.calculate_grad(problem_object, self.sigma_cons, w, t))
        grad_gs = np.asarray(self.calculate_grad(problem_object, self.tau_cons, w, t))
        grad_f = np.asarray(self.calculate_grad(problem_object, self.cost_function, w, t))

        A = np.array([grad_gd,
                      grad_gb,
                      grad_gs,
                      self.AL])
        
        b = np.array([np.dot(grad_gd, x) - self.displacement_cons(problem_object,x[0],x[1]),
                      np.dot(grad_gb, x) - self.sigma_cons(problem_object,x[0],x[1]),
                      np.dot(grad_gs, x) - self.tau_cons(problem_object,x[0],x[1]),
                      self.bL])

        # Bounds of the design veriables
        w_bounds = (opt.lb[0], opt.ub[0])
        t_bounds = (opt.lb[1], opt.ub[1])

        for i in range(self.max_iter):
            # Solve linear optimization problem using scipy.optimize.linprog
            res = linprog(grad_f, A, b, None, None, bounds=(w_bounds, t_bounds))
            x = res['x']
            constraints = np.array([self.displacement_cons(problem_object, x[0], x[1]),
                                    self.sigma_cons(problem_object, x[0], x[1]),
                                    self.tau_cons(problem_object, x[0], x[1]),
                                    np.dot(self.AL, x) - self.bL])

            print("iteration = {:<4d} ----- cost = {:<10f} ----- violation = {:<10f}"\
                .format(i+1, self.cost_function(problem_object,x[0],x[1]),
                        np.sum(constraints[constraints>0])))
            # Check if all constraints satisfy the converegence criterion
            if np.all(constraints <= self.tol):
                print('SLP terminated at iteration {:d}'.format(i+1))
                break
            
            # Constraint with maximum value of violation is selected to linearize 
            # about the new 'x' 
            max_violation_ind = np.argmax(constraints)

            # Add new constraint to previous ones.
            # Thus, a new Linear Programming problem is established and
            # is to be solved in the next iteration
            if max_violation_ind==0:
                grad_gd = np.asarray(self.calculate_grad(problem_object, self.displacement_cons, x[0], x[1]))
                A = np.append(A, [grad_gd], axis = 0)
                b = np.append(b, np.dot(grad_gd, x) - self.displacement_cons(problem_object, x[0], x[1]))
            if max_violation_ind==1:
                grad_gb = np.asarray(self.calculate_grad(problem_object, self.sigma_cons, x[0], x[1]))
                A = np.append(A, [grad_gb], axis = 0)
                b = np.append(b, np.dot(grad_gb, x) - self.sigma_cons(problem_object, x[0], x[1]))
            if max_violation_ind==2:
                grad_gs = np.asarray(self.calculate_grad(problem_object, self.tau_cons, x[0], x[1]))
                A = np.append(A, [grad_gs], axis = 0)
                b = np.append(b, np.dot(grad_gs, x) - self.tau_cons(problem_object, x[0], x[1]))
            if max_violation_ind==3:
                A = np.append(A, [self.AL], axis = 0)
                b = np.append(b, self.bL)

        print("w={:4f}, t={:4f}, Weight={:10f}".format(x[0], x[1], self.cost_function(problem_object,x[0],x[1])))    
        return x, self.cost_function(problem_object,x[0],x[1]), np.sum(constraints[constraints>0])
        
         
if __name__ == "__main__":
    pr = Problem()
    opt = SLPOptimization()
    x, f, viol = opt.run_SLP(pr, 100, 17)

