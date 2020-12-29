# COPYRIGHT
# Daniil Lyakhov
# dupeljan@gmail.com
import numpy as np
from math import ceil
from priorityQueue import PriorityQueue
from batchedModel import BatchedModel
from docplex.mp.model import Model
import threading

#from numba import jit
import sys

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
from core import trunc_precisely

# Hyperparameters
EPS = 1e-6
PRECISION = 4
INF = np.inf
INIT_COLORING_ATTEMPTS = 1# default 500 or 30
CLIQUE_HEURISTIC_ATTEMPTS = 300# default 300
SLAVE_SOLVER_TIMELIMIT = 1000000000000
SLAVE_HEURISTIC_ATTEMPTS = 10
problem_list = \
[
    # "anna.col", +
    # "david.col", +
    # "fpsol2.i.1.col",
    # "fpsol2.i.2.col",
    # "fpsol2.i.3.col",
    # "games120.col",
    # "homer.col",
    # "huck.col",
    # "inithx.i.1.col",
    # "inithx.i.2.col",
    # "inithx.i.3.col",
    # "jean.col",
    # "le450_15a.col",
    # "le450_15b.col",
    # "le450_15c.col",
    # "le450_15d.col",
    # "le450_25a.col",
    # "le450_25b.col",
    # "le450_25c.col",
    # "le450_25d.col",
    # "le450_5a.col",
    # "le450_5b.col",
    # "le450_5c.col",
    # "le450_5d.col",
    # "miles1000.col", +
    # "miles1500.col", +
    # "miles250.col",
    # "miles500.col", +
    # "miles750.col", +
    # "mulsol.i.1.col",
    # "mulsol.i.2.col",
    # "mulsol.i.3.col",
    # "mulsol.i.4.col",
    # "mulsol.i.5.col",
     # "myciel2.col", +
     # "myciel3.col", +
     # "myciel4.col",
     "myciel5.col",
     "myciel6.col",
     "myciel7.col",
    # "queen10_10.col",
    # "queen11_11.col",
    # "queen12_12.col",
    # "queen13_13.col",
    # "queen14_14.col",
    # "queen15_15.col",
    # "queen16_16.col",
     "queen5_5.col",
     "queen6_6.col",
     "queen7_7.col",
    # "queen8_12.col",
     "queen8_8.col",
    # "queen9_9.col",
    # "school1.col",
    # "school1_nsh.col",
    # "zeroin.i.1.col",
    # "zeroin.i.2.col",
    # "zeroin.i.3.col",
]

class MinColoringProblem(MaxCliqueProblem):
    """Class for MinColoringProblem solving
    by Branch and Price method.
    """
    @staticmethod
    def trunc_up(float_number):
        """wisely floor float number"""
        if np.isclose(float_number, round(float_number), atol=1e-6):
            truncated_number = round(float_number)
        else:
            truncated_number = ceil(float_number)
        return truncated_number

    def __init__(self, inp):
        super().__init__("coloring/" + inp, "COLORING")
        self.best_coloring_val = len(self.Nodes)
        self.best_coloring_set = self.Nodes
        self.master_contraints = dict()
        self.state_set_master_vars = list()
        self.forbiden_sets = set()
        self.branch_constraints = dict()
        #self.state_set_vars = StateSetConstraints()
        self.define_model_and_variables()
        self._precompute_for_slave_problem()
        self.history_branching = list()
        self._conf = True

    def _precompute_for_slave_problem(self):
        """Precompute constraints for slave problem"""
        self.slave_model = BatchedModel()
        # Add variables
        self.Y_slave = {n: self.slave_model.binary_var(name="y_{0}".format(n)) for n in self.Nodes}
        # Add ground constraints
        for i, j in self.Edges:
            self.slave_model.add_constraint_bath(self.Y_slave[i] + self.Y_slave[j] <= 1)
        # Add strong constraints
        #for i in range(CLIQUE_HEURISTIC_ATTEMPTS):
        #    clique = self.init_heuristic_clique(random=True)
        #    if len(clique) > 2:
        #        self.slave_model.add_constraint_bath(self.slave_model.sum([self.Y_slave[n] for n in clique]) <= 1)

    def ind_set_to_max_sorted_ind_set(self, set_):
        color_to_max_ind_set = self.maximal_ind_set_colors(set_)
        return {tuple(sorted(x)) for x in color_to_max_ind_set.values()}

    def update_best_value(self, color):
        """Update best color val and
        best val set by given color"""
        ind_set = self.colors_to_indep_set(color)
        if len(ind_set) < self.best_coloring_val:
            self.best_coloring_val = len(ind_set)
            self.best_coloring_set = list(ind_set.values())

    def get_variables(self, strategy='random_sequential'):
        color = nx.algorithms.coloring.greedy_color(self.G, strategy=strategy)
        self.update_best_value(color)
        return self.ind_set_to_max_sorted_ind_set(self.colors_to_indep_set(color))

    def _add_branch_constraint(self, constraint, i):
        """Add branch constraint to self.cp model
        params:
                constraint - given constraint"""
        assert i not in self.history_branching, "INF LOOP"
        self.master_model.add_constraint_bath(constraint)
        self.history_branching.append(i)


    def _remove_branch_constraint(self, constraint, i):
        """Remove branch constraint to self.cp model
        params:
                constraint - given constraint"""
        self.master_model.remove_constraint_bath(constraint)
        self.history_branching.remove(i)

    def update_master_model(self, state_set: tuple):
        ''' finction to update constraints, main variables and objective in the model '''
        assert state_set not in self.state_set_master_vars
        self.state_set_master_vars.append(state_set)

        self.X_mater_vars.update({len(self.X_mater_vars): self.master_model.continuous_var(name='x_{0}'.format(len(self.X_mater_vars)))})
        new_set = self.state_set_master_vars[-1]
        new_constr = []
        names = []
        for v in new_set:
            names.append(str(v))
            constr = self.master_model.get_constraint_by_name(str(v)).lhs\
                            + self.X_mater_vars[len(self.X_mater_vars) - 1] >= 1
            new_constr.append(constr)

        old_constraints = [self.master_model.get_constraint_by_name(str(v)) for v in new_set]
        self.master_model.remove_constraints(old_constraints)
        self.master_model.add_constraints(new_constr, names=names)
        self.master_model.apply_batch()
        # update objective
        self.master_model.remove_objective()
        self.master_model.minimize(self.master_model.sum([v for v in self.X_mater_vars.values()]))

    def define_model_and_variables(self, attempts=INIT_COLORING_ATTEMPTS):
        self.master_model = BatchedModel(name="Min_coloring")

        self.X_mater_vars = dict()
        vars = set()
        for _ in range(attempts):
            vars |= self.get_variables()

        # Add variables
        self.state_set_master_vars += list(vars)
        for i, state_set in enumerate(vars):
            self.X_mater_vars[i] = self.master_model.continuous_var(name='x_{0}'.format(i))
        # Init constraints
        names = []
        constraints = []
        for n in self.Nodes:
            names += [str(n)]
            constraint = [self.X_mater_vars[i]
                          for i, val in enumerate(self.state_set_master_vars)
                          if n in val]
            assert constraint
            constraints += [self.master_model.sum(constraint) >= 1]
        self.master_model.add_constraints(constraints, names=names)

        # Set objective
        self.master_model.minimize(self.master_model.sum(self.X_mater_vars))

    def solve(self, timeout=7200):
        self.timeout = timeout
        print("Best heuristic val:", self.best_coloring_val)
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
        _set = self.X_mater_vars[set_to_branch_index]
        val = self.master_model.solution.get_all_values()[set_to_branch_index]
        res = [_set == 0, _set == 1]
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
            # Set timelimit
            if timelimit != INF:
                self.slave_model.parameters.timelimit = timelimit
            else:
                self.slave_model.parameters.reset_all()
            # Add forbiden constraints
            # Set objective
            # HAVE TO CHECK ORDER
            self.slave_model.remove_objective()
            self.slave_model.maximize(self.master_model.sum([weights[i] * self.Y_slave[n] for i, n in enumerate(self.Nodes)]))
            # Solve problem
            sol = self.slave_model.solve()

            # remove forbiden constraints
            #self.m.remove_constraints(constr_forbid)

            if sol is not None:
                obj = sol.solve_details.best_bound#get_objective_value()
                if np.round(obj, PRECISION) > 1.:
                    val = sol.get_all_values()
                    res = self.ind_set_to_max_sorted_ind_set(
                        {0: tuple([n for i, n in enumerate(self.Nodes) if val[i] == 1.0])})
                    if res.copy().pop() in self.state_set_master_vars:
                        print("Duplicate constraints!")
                    return res.pop(), obj
                return (), obj
            return (), INF
        else:
            sep = self.separation(weights, local_search=True, use_default_weights=True)
            sep = {tuple(sorted(sep))} - self.forbiden_sets
            if not sep:
                return (), INF
            return sep.pop(),  INF


    def column_gerator_loop(self, solver=False, timelimit=SLAVE_SOLVER_TIMELIMIT,
                                                attempts=SLAVE_HEURISTIC_ATTEMPTS):
        """Trying to add some variables i.e. columns
         to prune computing branch
        params:
                solver: bool - use solver to column search with
                                given timeout = timelimit
                                else do heuristic given attempts times
                        timelimit - given timelimit
                        attempts - given attempts"""
        while True:
            weights = self.master_model.dual_values([self.master_model.get_constraint_by_name(str(v)) for v in self.Nodes])
            assert weights, "Weights is empty!"
            val_cur = self.master_model.solution.get_objective_value()
            cols, upper_bound, = self.column_generator(solver=solver,
                                                       weights=weights,
                                                       timelimit=timelimit,
                                                       attempts=attempts)
            if solver:
                assert cols not in self.forbiden_sets
            lower_bound = self.trunc_up(val_cur/upper_bound)
            if lower_bound >= self.best_coloring_val:
                print("Cut it!")
                return True

            if not cols or cols == {()}:
                return False

            self.update_master_model(cols)
            self.master_model.solve()

    def BnPColoring(self):
        # Check timeout
        if self.check_timeout():
            return

        sol = self.master_model.solve()
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
        val = self.master_model.solution.get_all_values()
        i = super().get_node_index_to_branch(val)

        # If we can't branch
        if i == -1:
            # Prove it by solver
            self.column_gerator_loop(solver=True, timelimit=INF)
            # Reprove it's integerness
            val = self.master_model.solution.get_all_values()
            if super().get_node_index_to_branch(val) == -1:
                sol = self.master_model.solution
                val = trunc_precisely(sol.get_objective_value())
                if val < self.best_coloring_val:
                    self.best_coloring_val = val
                    self.best_coloring_set = sol.get_all_values()
                    print("Found solution: ", self.best_coloring_val)

                return

        # Branch it
        for constr in self.get_optimal_branch_list(i):
            # Add constraint to model
            self._add_branch_constraint(constr, i)
            if constr.rhs.constant == 0:
                self.forbiden_sets |= set([self.state_set_master_vars[i]])
                forbid_set = self.state_set_master_vars[i]
                forb_constr = self.slave_model.add_constraint(self.slave_model.sum(
                                        [self.Y_slave[n] for n in forbid_set]) <= len(forbid_set) - 1)

            #print("Branch with constr: ", i)
            self.BnPColoring()
            self._remove_branch_constraint(constr, i)
            if constr.rhs.constant == 0:
                self.forbiden_sets -= set([self.state_set_master_vars[i]])
                self.slave_model.remove_constraint(forb_constr)

    def output_statistic(self, outp):
        outp.write("Problem INP: " + str(self.INP) + "\n")
        outp.write("Obj: " + str(self.best_coloring_val) + "\n")
        outp.write("Color sets: " + str([x for i, x in
                                         enumerate(self.state_set_vars.list_[:len(self.best_coloring_set)])
                                         if self.best_coloring_set[i] != 0.]) + "\n")
        outp.write("Time elapsed second: " + str(self.time_elapsed) + "\n")
        outp.write("Time elapsed minutes: " + str(self.time_elapsed / 60) + "\n")
        if self.is_timeout:
            outp.write("Stopped by timer\n")
        outp.write("\n")
        #outp.write(str(len([x for i, x in enumerate(self.state_set_vars.list_)
        #                                 if self.best_coloring_set[i] != 0.])))

if __name__ == '__main__':
    # Setup recursive stack size
    sys.setrecursionlimit(3000)
    #sys.stdout = open("dump.txt", "w")
    with open("min_coloring_results_auto.txt", "w") as out:
        for problem_name in problem_list:
            print("Start configure", problem_name)
            p = MinColoringProblem(problem_name)
            print("Start solving ", problem_name)
            p.solve()
            p.output_statistic(sys.stdout)
            p.output_statistic(out)

