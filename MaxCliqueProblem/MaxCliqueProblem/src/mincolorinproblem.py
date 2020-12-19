# COPYRIGHT
# Daniil Lyakhov
# dupeljan@gmail.com
import numpy as np
from priorityQueue import PriorityQueue
from batchedModel import BatchedModel
from docplex.mp.model import Model
import threading

#from numba import jit

from functools import reduce
import docplex.mp
#from docplex.mp.model import Model
from itertools import combinations as comb, cycle
from itertools import  count
from collections import namedtuple, deque
import networkx as nx
import time
import os
from core import MaxCliqueProblem

EPS = 1e-1
INF = np.inf
CLIQUE_HEURISTIC_ATTEMPTS = 20
problem_list = \
[
    "anna.col"
]

class MinColoringProblem(MaxCliqueProblem):
    """Class for MinColoringProblem solving
    by Branch and Price method.
    WARNING: add constraints only by
self.master_constraints += [constr]"""
    def __init__(self, inp):
        super().__init__("coloring/" + inp, "COLORING")
        self.define_model_and_variables()
        self.best_coloring_val = len(self.Nodes)
        self.best_coloring_set = self.Nodes
        self.forbiden_sets = set()
        self.master_contraints = dict()
        self.branch_constraints = dict()
        self._conf = True

    def get_variables(self, strategy='random_sequential'):
        color = nx.algorithms.coloring.greedy_color(self.G, strategy=strategy)
        color_to_max_ind_set = self.maximal_ind_set_colors(self.colors_to_indep_set(color))
        return {tuple(sorted(x)) for x in color_to_max_ind_set.values()}

    def _add_branch_constraint(self, constraint):
        """Add branch constraint to self.cp model
        params:
                constraint - given constraint"""
        self.cp.add_constraint_bath(constraint)

    def _remove_branch_constraint(self, constraint):
        """Remove branch constraint to self.cp model
        params:
                constraint - given constraint"""
        self.cp.remove_constraint_bath(constraint)

    def reload_constraints(self):
        # Remove all constraints first
        self.cp.remove_constraints([c for c in self.master_contraints])
        self.master_contraints = dict()
        # Recompute constraints
        for n in self.Nodes:
            contraint = []
            for i, v in enumerate(self.V):
                if n in v:
                    contraint += [self.X[i]]
            if contraint:
                constraint = self.cp.sum(contraint) >= 1
                self.master_contraints[n] = self.cp.add_constraint_bath(constraint)
        # Add it to model
        self.cp.add_constraint_bath(self.master_contraints)

    def add_variables(self, state_sets: set):
        """Add variables to model
        params: state_set: set - list of sets of nodes,
         which needs to be in model. state_sets must
         distinguish from all other variables"""
        shift = len(self.V)
        self.V |= state_sets
        for i, state_set in enumerate(state_sets):
            # Number in list for new var
            j = shift + i + 1
            self.X[j] = self.cp.continuous_var(name='x_{0}'.format(j))
        self.reload_constraints()

    def define_model_and_variables(self, attempts=50):
        self.cp = BatchedModel(name="Min_coloring")
        # Add variables
        # Variables now is state sets
        # Initialize model by some colors set
        self.V = set()
        for _ in range(attempts):
            self.V |= self.get_variables()
        # State set model vars
        self.X = {i: self.cp.continuous_var(name='x_{0}'.format(i)) for i in range(len(self.V))}
        # Nodes variables
        self.N_var = {i: self.cp.continuous_var(name="n_{0}".format(i)) for i in self.Nodes}
        # Load constraints
        self.reload_constraints()
        # Set objective
        self.cp.apply_batch()
        self.cp.minimize(self.cp.sum(self.X))

    def solve(self, timeout=7200):
        self.timeout = timeout
        assert self._conf, "Configurate model first"

        self.start_solve_with_timeout(self.BnPColoring, timeout)

    @staticmethod
    def is_tailing_off(val1, val2):
        """Return true if this two values are too close"""
        return np.abs(val1 - val2) < EPS

    def get_optimal_branch_list(self, set_to_branch_index: int):
        """Return branch constraints in optimal order
        params:
                set_to_branch_index: int - index of branchin set
                                        in self.X
        returns:
                list of two constraints:
                [self.X[i] <= 0, self.X[i] >= 1]
                or reversed one where
                i = set_to_branch_index"""
        _set = self.X[set_to_branch_index]
        val = self.cp.solution.get_all_values()[set_to_branch_index]
        res = [_set <= 0, _set >= 1]
        return res if val < 0.5 else res[::-1]

    def column_generator(self, weights, solver=False, timelimit=10, attempts=10):
        """Try to find violated constraint in dual problem
        which is equal to variable search in master problem
        Problem is: Find heaviest state set with given
        weights
        params:
                weights: given weights
                solver: bool - use solver if it's true
                                else use heuristic
                timelimit: int - timelimit for solver
                attempts: int - attempts to find solution for
                                heuristic
        return:
                (columns, upper_bound)
                where
                columns - set of state sets which violate
                constraints
                upper_bound - upper_bound for this task"""
        if solver:
            # Define model and set timelimit
            m = BatchedModel()
            m.parameters.timelimit = timelimit

            # Add variables
            Y = {n: m.binary_var(name="y_{0}".format(n)) for n in self.Nodes}
            # Add ground constraints
            for i, j in comb(self.Nodes, 2):
                if [i, j] in self.Edges and [j, i] in self.Edges:
                    m.add_constraint_bath([Y[i] + Y[j] <= 1])
            # Add strong constraints
            for i in range(CLIQUE_HEURISTIC_ATTEMPTS):
                clique = self.init_heuristic_clique(random=True)
                m.add_constraint_bath(m.sum([Y[n] for n in clique] <= 1))
            # Add forbiden constraints
            for forbid_set in self.forbiden_sets:
                m.add_constraint_bath(m.sum([Y[n] for n in forbid_set]) <= len(forbid_set) - 1)
            # Set objective
            # HAVE TO CHECK ORDER
            m.maximize(self.cp.sum([weights[n]*Y[n] for n in self.Nodes]))
            # Solve problem
            sol = m.solve()
            if sol is not None:
                obj = sol.get_objective_value()
                if obj > 1.:
                    val = sol.get_all_values()
                    return self.maximal_ind_set_colors(self.colors_to_indep_set(val)), val
            return {}, INF
        else:
            several_sep = self.several_separation(weights, count=attempts, local_search=True)
            return several_sep - self.forbiden_sets, INF

    def column_gerator_loop(self, solver=False, timelimit=10, attempts=10):
        """Trying to add some variables i.e. columns
         to prune computing branch
        params:
                solver: bool - use solver to column search with
                                given timeout = timelimit
                                else do heuristic given attempts times
                        timelimit - given timelimit
                        attempts - given attempts"""
        while True:
            weights = self.cp.dual_values(self.master_contraints.values())
            val_cur = self.cp.solution.get_all_values()
            cols, upper_bound, = self.column_generator(solver=solver,
                                                       weights=weights,
                                                       timelimit=timelimit,
                                                       attempts=attempts)

            lower_bound = np.round(0.5 + np.sum(weights)/upper_bound)
            if lower_bound > self.best_colorign_val:
                return True

            if not cols:
                return False

            self.add_variables(cols)
            sol = self.cp.solve()
            # Exit if solution doesn't change
            # significantly
            if MinColoringProblem.is_tailing_off(sol.get_all_values(), val_cur):
                return False

    def BnPColoring(self):
        # Check timeout
        if self.check_timeout():
            return

        sol = self.cp.solve()

        # Prune if there is no solution
        # on this branch
        if sol is None:
            return

        # Try to add new valuable variables
        # and maybe prune this branch
        for solver in [False, True]:
            can_prune_branch = self.column_gerator_loop(solver=solver)
            if can_prune_branch:
                return

        # If we can't prune it
        # thus try to branch it
        i = super().get_node_index_to_branch()

        # If we can't branch
        if i == -1:
            # Prove it by solver
            self.column_gerator_loop(solver=True, timelimit=INF)
            # Reprove it's integerness
            if super().get_node_index_to_branch() == -1:
                sol = self.cp.solution
                self.best_coloring_val = sol.get_objective_value()
                self.best_coloring_set = sol.get_all_values()

            return

        # Branch it
        for constr in self.get_optimal_branch_list(i):
            # Add constraint to model
            self._add_branch_constraint(constr)
            if constr.rhs == 0:
                self.forbiden_sets |= set(constr.lhs)

            self.BnPColoring()

            self._remove_branch_constraint(constr)
            if constr.rhs == 0:
                self.forbiden_sets -= set(constr.lhs)


if __name__ == '__main__':
    for problem in problem_list:
        p = MinColoringProblem(problem)
        p.solve()