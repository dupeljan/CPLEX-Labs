# COPYRIGHT
# Daniil Lyakhov
# dupeljan@gmail.com
import numpy as np
from priorityQueue import PriorityQueue
from batchedModel import BatchedModel

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

problem_list = \
[
    "anna.col"
]
class MinColoringProblem(MaxCliqueProblem):

    def __init__(self, inp):
        super().__init__("coloring/" + inp, "COLORING")
        self._configure()

    def get_variables(self, strategy='random_sequential'):
        color = nx.algorithms.coloring.greedy_color(self.G, strategy=strategy)
        color_to_max_ind_set = self.maximal_ind_set_colors(self.colors_to_indep_set(color))
        return {tuple(sorted(x)) for x in color_to_max_ind_set.values()}

    def load_constraints(self):
        self.master_contraints = dict()
        for n in self.Nodes:
            contraint = []
            for i, v in enumerate(self.V):
                if n in v:
                    contraint += [self.X[i]]
            if contraint:
                constraint = self.cp.sum(contraint) >= 1
                self.master_contraints[n] = self.cp.add_constraint_bath(constraint)

    def add_variables(self, state_sets: set):
        """Add variables to model
        params: state_set: set - list of sets of nodes,
         which """
        print(state_sets)

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
        self.load_constraints()
        # Set objective
        self.cp.apply_batch()
        self.cp.minimize(self.cp.sum(self.X))


    def _configure(self):
        self.define_model_and_variables()
        self._conf = True

    def solve(self, timeout=7200):
        self.timeout = timeout
        assert self._conf, "Configurate model first"

        self.start_solve_with_timeout(self.BnPColoring, timeout)


    def BnPColoring(self):
        # Check timeout
        if self.check_timeout():
            return

        # Try to find new stable sets
        while True:
            # Solve the problem
            # in current state
            sol = self.cp.solve()

            # If there is no solution
            #if sol is None:
            #    return

            obj = sol.get_objective_value()
            val = sol.get_all_values()

            weights = self.cp.dual_values(self.master_contraints.values())
            several_sep = self.several_separation(weights, count=5)
            self.add_variables(several_sep)

            i = super().get_node_index_to_branch(val)
            if several_sep:
                pass



if __name__ == '__main__':
    for problem in problem_list:
        p = MinColoringProblem(problem)
        p.solve()