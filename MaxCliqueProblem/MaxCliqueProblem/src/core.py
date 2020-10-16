import numpy as np
from priorityQueue import PriorityQueue

import docplex.mp
from docplex.mp.model import Model
from itertools import combinations as comb
from itertools import  count
from collections import namedtuple, deque
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

    def init_heuristic(self):
        """Try to find max clique
        use heuristic neighbors method
        """
        print("Start heuristic search...")
        for v in self.Nodes:
            # Add current node to
            # current clique
            clique_cur = {v}
            clique_neighbors = set(self.G.neighbors(v))

            while True:
                # Compute neighbors interception
                for c in clique_cur:
                    clique_neighbors &= set(self.G.neighbors(c))

                # Elements in clique can't be
                # clique neighbors
                clique_neighbors -= clique_cur

                # Exit if we can't add anything
                if not clique_neighbors:
                    break

                # Find best candidate
                candidates_deg = np.array(self.G.degree(clique_neighbors))
                i = np.argmax(candidates_deg[:, 1])
                candidate_best = candidates_deg[i][0]

                # Add it to clique
                clique_cur |= {candidate_best}

            # Keep new clique
            # if it's better than previous
            if len(clique_cur) >= self.objective_best:
                self.objective_best = len(clique_cur)
                self.objective_best_vals = clique_cur
                print("-------------Find solution: ", self.objective_best, "-------------")



    def colors_to_indep_set(self, coloring):
        '''Return dict, where
        key is collor and
        value is list of nodes
            colored in key color
        '''
        comp = dict()
        for vert, color in coloring.items():
            if color not in comp:
                comp[color] = []
            comp[color] += [vert]
        return comp

    def maximal_ind_set_colors(self, ind_set):
        """Maximaze independent set
        for each color in ind_set
        """
        ind_set_maximal = {i: set(v) for i, v in ind_set.items()}
        # Choose pairs of colors
        for i in ind_set.keys():
            for j, color_find in ind_set.items():
                if i != j:
                    # Choose elem from color_new
                    for x in color_find:
                        # If you can add it to
                        # this independent set
                        if all([not self.G.has_edge(x, y) for y in ind_set_maximal[i]]):
                            ind_set_maximal[i].add(x)

        return ind_set_maximal

    def __init__(self):
        self._conf = False
        self.get_input()
        self.G = nx.Graph()
        self.objective_best = 0
        self.objective_best_vals = 0
        print("Start conf model")
        self.configure_model()
        print("End conf model")
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
        self.BnBMaxCliqueDFS()
        self.time_elapsed = time.time() - start_time

    def get_input(self):
        INP = ['c125.9.txt', 'keller4.txt', 'p_hat300_1.txt', 'brock200_2.txt'][1]
        self.Edges = [list(map( int, str_.split()[1:3])) for str_ in open('input/'+INP).readlines() if str_[0] == 'e']
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
        bath = []
        for i, j in comb(self.Nodes, 2):
            if [i, j] not in self.Edges and [j, i] not in self.Edges:
                bath += [self.Y[i] + self.Y[j] <= 1]

        self.cp.add_constraints(bath)

        # Add constrains: nodes cant be in one clique
        # if you can color them in one color
        # Create graph

        self.G.add_edges_from(self.Edges)

        comps = set()
        # Color it
        for strategy in ['largest_first', 'smallest_last',
                         'independent_set', 'connected_sequential_bfs',
                         'connected_sequential_dfs', 'saturation_largest_first']:


            coloring = nx.algorithms.coloring.greedy_color(self.G, strategy=strategy)


            ind_set = self.colors_to_indep_set(coloring)
            # Get it to maximal on including

            ind_set_maximal = self.maximal_ind_set_colors(ind_set)

            comps |= {self.cp.sum([self.Y[i] for i in x]) <= 1 for x in ind_set_maximal.values()}

        # Trying find best coloring in randomized way
        coloring_list = list()
        for i in range(100):
            coloring_list.append(self.colors_to_indep_set(nx.algorithms.coloring.greedy_color(self.G,
                                                                                        strategy='random_sequential')))
        coloring_list = sorted(coloring_list, key=lambda x: len(x.keys()))

        for elem in coloring_list[:5]:
            ind_set_maximal = self.maximal_ind_set_colors(elem)
            comps |= {self.cp.sum([self.Y[i] for i in x]) <= 1 for x in ind_set_maximal.values()}

        # Add constraints
        self.cp.add_constraints(comps)

        # Set objective
        self.cp.maximize(self.cp.sum(self.Y))

        # Try to find heuristic solution
        self.init_heuristic()

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

        # Get best branching value
        i = MaxCliqueProblem.get_node_index_to_branch(val)
        # If current solution better then previous
        if i == -1:
            if obj > self.objective_best:
                print("--------------------Find solution: ", obj, '--------------------')
                # Remember it
                self.objective_best = obj
                self.objective_best_vals = val
            return

        # Else - branching

        for ind in [0, 1]:
            # Set i-th constraint to val
            constr = self.cp.add_constraint(self.Y[self.Nodes[i]] == ind)

            print("Branch: " + str(constr), "obj: " + str(obj))

            # Recursive call
            self.BnBMaxClique()

            # Remove constrain from the model
            self.cp.remove_constraint(constr)

    def BnBMaxCliqueDFS(self):
        """Compute optimal solutuon
        for max clique problem using
        Branch and bounds method.
        Use DFS during computations
        """
        # CP must be CPLEX model
        # inicialized for MaxClique
        # problem
        assert self._conf, "Configurate model first!"

        stack = deque()

        stack.append({"val": 0})
        cons = self.cp.number_of_constraints
        cons_before = 0
        while stack:
            elem = stack.pop()
            # If we don't visit this node yet
            if elem["val"] == 0:
                sol = self.cp.solve()
                if sol is None:
                    print("None solution found")
                    continue

                obj = sol.get_objective_value()

                # Cut if current node upper bound
                # less than best int obj
                if trunc_precisely(obj) <= self.objective_best:
                    self.cutted += 1
                    print("Cut branch: ", self.cutted)
                    continue

                vals = sol.get_all_values()
                i = MaxCliqueProblem.get_node_index_to_branch(vals)

                # If it's integer solution
                if i == -1:
                    # If current solution better then previous
                    if self.objective_best < obj:
                        print("--------------------Find solution: ", obj, '--------------------')
                        # Remember it
                        self.objective_best = obj
                        self.objective_best_vals = vals

                    continue

                # Set elem branch var
                i = self.Y[self.Nodes[i]]
                elem['constr'] = self.cp.add_constraint(i == 0)

                print("Branch by ", i, " obj: ", obj)
                for ind in range(2):
                    stack.append(elem)
                    stack.append({"val": 0})


                '''
                cons_new = self.cp.number_of_constraints - cons
                print("0N ", cons_new)
                if cons_before + 1 != cons_new:
                    print("0ERROR")
                cons_before = cons_new
                '''
                
                elem["val"] += 1

            # If it's already visited once node
            elif elem["val"] == 1:
                constr = elem["constr"]
                self.cp.remove_constraint(constr)
                elem["constr"] = self.cp.add_constraint(constr.lhs == 1)

                '''
                cons_new = self.cp.number_of_constraints - cons
                print("1N ", cons_new)
                if cons_before != cons_new:
                    print("1ERROR")
                cons_before = cons_new
                '''

                elem["val"] += 1

            # If it's already visited twice node
            else:
                self.cp.remove_constraint(elem["constr"])

                '''
                cons_new = self.cp.number_of_constraints - cons
                print("2N ", cons_new)
                if cons_before - 1 != cons_new:
                    print("2ERROR")
                cons_before = cons_new

                print("2N ", self.cp.number_of_constraints - cons)
                '''

    def BnBMaxCliqueBFS(self):
        """Compute optimal solutuon
        for max clique problem using
        Branch and bounds method.
        Go thought maximum bound nodes during
        computation
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
        unique_constr = list()
        while True:
            # Pop node from queue
            pop = nodes.pop_task_and_priority()
            if pop is None:
                break
            obj, node = pop

            constr_set = node.constraints
            print("obj : ", obj)

            # Get constrains intersection
            time_beg = time.time()
            intersec = constr_set_prev.intersection(constr_set)
            # Remove inappropriate constraints

            self.cp.remove_constraints(constr_set_prev.difference(intersec))
            # Add appropriate constraints
            self.cp.add_constraints(constr_set.difference(intersec))

            constr_set_prev = constr_set
            print("Time to jump: ", time.time() - time_beg)
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
               if self.objective_best < obj:
                    print("--------------------Find solution: ", obj, '--------------------')
                    self.objective_best = obj
                    self.objective_best_vals = constr_set
            
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