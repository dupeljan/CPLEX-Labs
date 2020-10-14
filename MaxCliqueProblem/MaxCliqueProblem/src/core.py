import numpy as np
from priorityQueue import PriorityQueue

import docplex.mp
from docplex.mp.model import Model
from itertools import combinations as comb
from itertools import  count
from collections import namedtuple
import networkx as nx
import time



EPS = 1e-8


def is_int(elem):
    return np.isclose(elem, np.round(elem), atol=EPS)


def is_int_list(elems):
    """Return true if
    all elements in iterable arg elems
    are close to integers
    """
    res = True
    for elem in elems:
        if not np.isclose(elem, np.round(elem), atol=EPS):
            res = False
            break
    return res

def trunc_precisely(elem):
    """Trancate value in responce to
    EPS precision
    """
    return  np.round(elem) if is_int(elem) else np.trunc(elem)

class MaxCliqueProblem:

    Node = namedtuple('Node', ['constraints', 'var_to_branch'])


    def __init__(self):
        self._conf = False
        self.get_input()
        print("Start conf model")
        self.configure_model()
        print("End conf model")
        self.objective_best = 0
        self.objective_best_vals = 0
        self.cutted = 0
        self.time_elapsed = 0
    
    def solve(self):
        # CP must be CPLEX model
        # inicialized for MaxClique
        # problem
        assert self._conf, "Configurate model first!"
        print("Start to solve")
        #self.BnBMaxClique()
        start_time = time.time()
        self.BnBMaxCliqueNonRecursive()
        self.time_elapsed = time.time() - start_time

    def get_input(self):
        INP = ['c125.9.txt','keller4.txt','p_hat300_1.txt','brock200_2.txt'][0]
        self.Edges = [ list(map( int, str_.split()[1:3])) for str_ in open('input/'+INP).readlines() if str_[0] == 'e' ]
        self.Nodes = list(set([ y for x in self.Edges for y in x]))
        # Set variable to protect BnB metod from unconfigurate model
        self._conf = False



    def configure_model(self):
        self.cp = Model(name='Max_clique')

        # Continious model vars 
        self.Y = {i: self.cp.continuous_var(name='y_{0}'.format(i)) for i in self.Nodes}

        # y constrains
        self.Y_cons = {n: self.cp.add_constraint(self.Y[n] <= 1) for n in self.Nodes}

        # Constrains for clique
        # Natural constraints
        for i, j in comb(self.Nodes,2):
            if [i, j] not in self.Edges and [j, i] not in self.Edges:
                self.cp.add_constraint((self.Y[i] + self.Y[j]) <= 1)

        # Add constrains: nodes cant be in one clique
        # if you can color them in one color
        # Create graph
        G = nx.Graph()
        G.add_edges_from(self.Edges)
        # Color it
        coloring = nx.algorithms.coloring.greedy_color(G)
        # Get components of one color
        comp = dict()
        for vert, color in coloring.items():
            if color not in comp:
                comp[color] = []
            comp[color] += [self.Y[vert]]

        # Add constraints
        for c in comp.values():
            self.cp.add_constraint(self.cp.sum(c) <= 1)



        # Set objective
        self.cp.maximize(self.cp.sum(self.Y))
        # Allow BnB to work
        self._conf = True

    @staticmethod
    def get_node_index_to_branch(elems):
        """Choose most apropriate 
        elem from elems to make 
        the branch.
        Return index of most appropriate element
        from list or -1 if all elements is integers
        """
        i = -1
        val = 1.0
        for j, elem in enumerate(elems):
            val_ = abs(elem - np.round(elem))
            if EPS < val_ < val:
                i = j
                val = val_
                
        return i
                

    def BnBMaxClique(self):
        """Compute optimal solutuon
         for max clique problem using
         Branch and bounds method
         """

        # Solve the problem
        # in current state
        sol = self.cp.solve()

        # If there is no solution
        if sol is None:
            print("None solution found")
            # Cut it
            return

        obj = sol.get_objective_value()
        val = sol.get_all_values()

        # If current branch upper bound not more
        # than known objective
        if trunc_precisely(obj) <=  self.objective_best:
            self.cutted += 1
            print("Cut branch. Obj: ", obj, " Cutting count: ", self.cutted)
            # Cut it
            return

        # If current solution better then previous
        if is_int_list(val) and obj > self.objective_best:
            print("--------------------Find solution: ", obj, '--------------------')
            # Remember it
            self.objective_best = obj
            self.objective_best_vals = val

        # Else - branching

        # Get best branching value
        i = MaxCliqueProblem.get_node_index_to_branch(val)

        for ind in [0, 1]:
            # Set i-th constraint to val
            constr = self.cp.add_constraint(self.Y[self.Nodes[i]] == ind)

            print("Branch: " + str(constr), "obj: " + str(obj))

            # Recursive call
            self.BnBMaxClique()

            # Remove constrain from the model
            self.cp.remove_constraint(constr)

    def BnBMaxCliqueNonRecursive(self):
        """Compute optimal solutuon
        for max clique problem using
        Branch and bounds method 
        """
        # CP must be CPLEX model
        # inicialized for MaxClique
        # problem
        assert self._conf, "Configurate model first!"
        
        nodes = PriorityQueue()
        sol = self.cp.solve()
        objective_best = (0, 0)
        # Constraints of pred computed node
        constr_set_prev = frozenset()
        variables_set = set(self.Y.values())
        obj = sol.get_objective_value()
        i = MaxCliqueProblem.get_node_index_to_branch(sol.get_all_values())
        var_to_branch_new = self.Y[self.Nodes[i]] if i != -1 else None
        
        # Put solution into queue.
        # And put beginning node into tree
        nodes.add_task(priority=obj, task=MaxCliqueProblem.Node(constraints=frozenset(),
                                                                var_to_branch=var_to_branch_new))
        
        while nodes:
            # Pop node from queue
            obj, node = nodes.pop_task_and_priority()

            constr_set = node.constraints
            print("obj : ", obj)

            # Get constrains intersection
            intersec = constr_set_prev.intersection(constr_set)
            # TODO use bath constraints
            # Remove inappropriate constraints
            for x in constr_set_prev.difference(intersec):
                self.cp.remove_constraint(x)
            # Add appropriate constraints
            for x in constr_set.difference(intersec):
                self.cp.add_constraint(x)

            constr_set_prev = constr_set
            # Apply 

            # Cut if current node upper bound
            # less than best int obj
            if trunc_precisely(obj) <= self.objective_best:
                self.cutted += 1
                print("Cut branch. Obj: ", obj, " Cutting count: ", self.cutted)
                # Cut it
                continue

            # Get variable to branch

            var_to_branch = node.var_to_branch
            #
            #var_to_branch = variables_set.difference({x.left_expr for x in constr_set}).pop()
            #i = MaxCliqueProblem.get_node_index_to_branch(self.cp.)

            # If it's integer solution
            if var_to_branch is None:

               # Remember best solution
               if objective_best[0] < obj:
                    print("--------------------Find solution: ", obj, '--------------------')
                    objective_best = (obj, constr_set)
            
            else:

                # Branch it!

                for ind in [0, 1]:
                    # Set i-th constraint to val 
                    constr = self.cp.add_constraint(var_to_branch == ind)

                    # Solve it
                    sol = self.cp.solve()

                    print("Branching: ", constr)
                    if sol is not None:
                        # Save results
                        obj = sol.get_objective_value()
                        # Find appropriate variable to branch
                        i = MaxCliqueProblem.get_node_index_to_branch(sol.get_all_values())
                        var_to_branch_new = self.Y[self.Nodes[i]] if i != -1 else None
                        node_new = MaxCliqueProblem.Node(constraints=constr_set.union({constr}),
                                                         var_to_branch=var_to_branch_new)
                        nodes.add_task(priority=obj, task=node_new)
                    
                    # Remove constrain from the model
                    self.cp.remove_constraint(constr)

                
if __name__ == "__main__":
    import pathlib
    print(pathlib.Path(__file__).parent.absolute())
    problem = MaxCliqueProblem()
    problem.solve()
    print(problem.objective_best)
    print(problem.objective_best_vals)
    print("Time elapsed: ", problem.time_elapsed)