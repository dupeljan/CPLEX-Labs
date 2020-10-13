import numpy as np
from priorityQueue import PriorityQueue

import docplex.mp
from docplex.mp.model import Model
from itertools import combinations as comb
from itertools import  count
from dataclasses import dataclass
import networkx as nx



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

    @dataclass
    class Node:
        """Specify structure
        for nodes
        """

    def __init__(self):
        self._conf = False
        self.get_input()
        print("Start conf model")
        self.configure_model()
        print("End conf model")
        self.best_objective = 0
        self.best_objective_vals = 0
        self.cutted = 0
    
    def solve(self):
        # CP must be CPLEX model
        # inicialized for MaxClique
        # problem
        assert self._conf, "Configurate model first!"
        print("Start to solve")
        self.BnBMaxClique()

    def get_input(self):
        INP = ['c125.9.txt','keller4.txt','p_hat300_1.txt','brock200_2.txt'][0]
        self.Edges = [ list(map( int, str_.split()[1:3])) for str_ in open('input/'+INP).readlines() if str_[0] == 'e' ]
        self.Nodes = list(set([ y for x in self.Edges for y in x]))
        # Set variable to protect BnB metod from unconfigurate model
        self._conf = False



    def configure_model(self):
        self.cp = Model(name='Max_clique')

        # Continious model vars 
        self.Y = {i : self.cp.continuous_var(name= 'y_{0}'.format(i)) for i in self.Nodes}

        # y constrains
        self.Y_cons = { n : self.cp.add_constraint( self.Y[n] <= 1) for n in self.Nodes }

        # Constrains for clique
        # Natural constraints
        for i,j in comb(self.Nodes,2):
            if [i,j] not in self.Edges and [j,i] not in self.Edges:
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
        if trunc_precisely(obj) <=  self.best_objective:
            self.cutted += 1
            print("Cut branch. Obj: ", obj, " Cutting count: ", self.cutted)
            # Cut it
            return

        # If current solution better then previous
        if is_int_list(val) and obj > self.best_objective:
            print("--------------------Find solution: ", obj,'--------------------')
            # Remember it
            self.best_objective = obj
            self.best_objective_vals = val

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
        best_objective = (0,0)
        id = 0
        # Id of pred computed node
        prev = 0

        comp_tree = nx.Graph()
        obj = sol.get_objective_value()
        values = sol.get_all_values()
        
        # Put solution into queue.
        # And put beginning node into tree
        nodes.add_task(priority=obj, task=id)
        comp_tree.add_node(id)
        id += 1
        
        while nodes:
            # Pop node from queue
            obj, cur_id = nodes.pop_task_and_priority()

            # Get Path of constraints
            constr_path = nx.single_source_shortest_path(comp_tree)[prev]

            # Apply 


            # Cut if current node upper bound
            # less than best int obj
            if trunc_precisely(obj) <=  self.best_objective:
                self.cutted += 1
                print("Cut branch. Obj: ", obj, " Cutting count: ", self.cutted)
                # Cut it
                continue

            print("obj : ", obj)
            print("task", task)
            
            # If it's integer solution
            if i == -1:
               # Remember best solution
               if best_objective[0] < obj:
                    best_objective = (obj,values)
            
            else:
                # Branch it!

                i = MaxCliqueProblem.get_node_index_to_branch(values)
                    
                for ind in [0,1]:
                    # Set i-th constraint to val 
                    constr = self.cp.add_constraint(self.Y[i] == ind)
                    
                    # Solve it
                    sol = self.cp.solve()

                    if sol is not None:
                        # Save results
                        obj = sol.get_objective_value()
                        values = sol.get_all_values()
                        nodes.add_task(priority=obj, task={ 'values' :values, 'constraints' : constraints + [constr] })
                    
                    # Remove constrain from the model
                    self.cp.remove_constraint(constr)

                
if __name__ == "__main__":
    import pathlib
    print(pathlib.Path(__file__).parent.absolute())
    problem = MaxCliqueProblem()
    problem.solve()
    print(problem.best_objective)
    print(problem.best_objective_vals)