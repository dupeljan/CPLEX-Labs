# COPYRIGHT
# Daniil Lyakhov
# dupeljan@gmail.com
import numpy as np
from priorityQueue import PriorityQueue
from batchedModel import BatchedModel

import threading

from functools import  reduce
import docplex.mp
#from docplex.mp.model import Model
from itertools import combinations as comb, cycle
from itertools import  count
from collections import namedtuple, deque
import networkx as nx
import time
from multiprocessing import Process


EPS = 1e-8

Content_list = \
        [
    #    'c-fat200-1.clq',
   #     'c-fat200-2.clq',
   #     'c-fat200-5.clq',
   #     'c-fat500-1.clq',
  #      'c-fat500-10.clq',
  #      'c-fat500-2.clq',
  #      'c-fat500-5.clq',
  #      'MANN_a9.clq',
  #      'hamming6-2.clq',
 #       'hamming6-4.clq',
 #       'gen200_p0.9_44.clq',
 #       'gen200_p0.9_55.clq',
#        'san200_0.7_1.clq',
 #       'san200_0.7_2.clq',
 #       'san200_0.9_1.clq',
 #       'san200_0.9_2.clq',
#        'san200_0.9_3.clq',
 #       'sanr200_0.7.clq', Must run ones again
 #       'C125.9.clq',
#       'keller4.clq',
        #'brock200_1.clq',
#        'brock200_2.clq',
        'brock200_3.clq',
        'brock200_4.clq',
#        'p_hat300-1.clq',
        'p_hat300-2.clq'
        ]






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
    round_elem = np.round(elem)
    return round_elem if np.isclose(elem, round_elem, atol=EPS) else np.trunc(elem)

class MaxCliqueProblem:

    Node = namedtuple('Node', ['constraints', 'var_to_branch'])

    def _is_best_vas_clique(self):
        """Test if self.best_obj_vals is clique"""
        for n in self.objective_best_vals:
            neighbors = list(self.G.neighbors(n))
            for j in set(self.objective_best_vals) - {n}:
                if j not in neighbors:
                    return False
        return True

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

    def local_search(self, clique_inp= None):
        """Try to find better solution
        based in best known solution.
        Solution must be set of nodes"""

        class Clique:
            """Incapsulate clique manipulation"""
            def __init__(self, clique, G):
                # tau[n] =
                # edge count from clique neighbor n
                # to clique
                self.G = G
                self.tau = {n: 0 for n in self.G.nodes}
                self.clique = set(clique)
                self.clique_list = []
                self.neighbors = reduce(lambda x, y: x | y, [set(self.G.neighbors(x)) for x in self.clique]) - \
                                                                                                    self.clique
                self._initzialized = False
                self.maximize_init()
                self._initzialized = True
                self.iter_index = 0


            def delete(self, value):
                """Delete value from the
                clique and update neighbors"""
                self.clique -= {value}
                self.tau[value] = len(self.clique) - 1
                # Delete neighbors which is connected
                # only with value
                neighbors = list(self.G.neighbors(value))
                neighbors_value_only = {n for n in neighbors if self.tau[n] == 1}
                self.neighbors -= set(neighbors_value_only)
                for n in neighbors:
                    self.tau[n] -= 1

                # rewind iterations
                self._rewind()

            def add(self, values):
                """Add values to clique
                and update neighbors and tau.
                    values must be a set!"""
                # Add values to list
                self.clique |= values
                # Update neighbors
                new_neighbors = reduce(lambda x, y: x | y, [set(self.G.neighbors(x)) for x in values])
                self.neighbors |= new_neighbors
                # Remove clique nodes from neighbors
                self.neighbors -= self.clique
                # Update tau if already initialized
                if self._initzialized:
                    for n in new_neighbors:
                        self.tau[n] += 1
                        # if tau is equal clique len
                        # then we need to add this vertex to clique
                        if self.tau[n] == len(self.clique):
                            self.add({n})

                # rewind iterations
                self._rewind()

            def maximize_init(self):
                """Try to maximize current clique
                just find appropriate node in the
                list of the neighbors"""
                repeat = True
                while repeat:
                    repeat = False

                    for n in self.neighbors:
                        b = len(set(self.G.neighbors(n)) & self.clique)
                        self.tau[n] = b
                        # If we can add n to clique
                        if b == len(self.clique):
                            # add it and repeat from the begining
                            self.add({n})
                            repeat = True
                            break

            def _rewind(self):
                """Start iteration from the
                begining"""
                self.clique_list = list(self.clique)
                self.iter_index = 0

            def __iter__(self):
                self.clique_list = list(self.clique)
                return self

            def __next__(self):
                try:
                    self.iter_index += 1
                    return self.clique_list[self.iter_index - 1]
                except IndexError:
                    raise StopIteration

            def __len__(self):
                return len(self.clique)

        if clique_inp is None:
            clique_inp = self.objective_best_vals

        clique = Clique(set(clique_inp), self.G)
        assert self._is_best_vas_clique(), "ERROR"
        swap = ()
        for x in clique:

            # Find candidates to add
            candidates = {n for n in clique.neighbors - set(self.G.neighbors(x)) if clique.tau[n] == len(clique) - 1}
            for c in candidates:
                # TODO sort and use set instead of set
                pair = set(self.G.neighbors(c)) & candidates
                # If we can add 2 nodes instead of one
                if pair:
                    swap = (x, c, pair.pop())
                    print("Improve solution by 1!")
                    # Update clique

                    clique.delete(swap[0])
                    clique.add(set(swap[1:]))
                    self.objective_best_vals = clique.clique
                    assert self._is_best_vas_clique(), "ERROR"
                    break

        if len(clique.clique) > self.objective_best:
            print("---------------------------Find new solution: ", len(clique.clique),
                                    " by local search---------------------------")
            self.objective_best = len(clique.clique)
            self.objective_best_vals = clique.clique


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

    def __init__(self, inp):
        self.INP = inp
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
        self.is_timeout = False

    def solve(self, timeout=7200):
        # CP must be CPLEX model
        # inicialized for MaxClique
        # problem
        self.timeout = timeout
        assert self._conf, "Configurate model first!"
        # Try to find heuristic solution
        self.init_heuristic()
        assert self._is_best_vas_clique(), "ERROR"
        print("Perform local search..")
        self.local_search()
        print("Start to solve")
        #self.BnBMaxClique()
        # Limit solving on time
        self.start_time = time.time()
        self.BnBMaxCliqueDFS()
        self.time_elapsed = time.time() - self.start_time


    def get_input(self):

        #self.INP = ['c125.9.txt', 'keller4.txt', 'p_hat300_1.txt', 'brock200_2.txt'][self.inp_no]
        self.Edges = [list(map( int, str_.split()[1:3])) for str_ in open('input/DIMACS_all_ascii/'+self.INP).readlines() if str_[0] == 'e']
        self.Nodes = list(set([ y for x in self.Edges for y in x]))
        # Set variable to protect BnB metod from unconfigurate model
        self._conf = False

    def configure_model(self):
        self.cp = BatchedModel(name='Max_clique')

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

        # Allow BnB to work
        self._conf = True

    @staticmethod
    def get_node_index_to_branch(elems: np.array):
        """Choose most apropriate 
        elem from elems to make 
        the branch.
        Return index of most appropriate element
        from self.Node list or -1 if all elements is integers
        """

        dists_to_near_int = np.abs(elems - np.round(elems))
        candidates = dists_to_near_int[dists_to_near_int >= EPS]
        if not np.any(candidates):
            return -1
        return np.argwhere(dists_to_near_int == np.min(candidates)).reshape(-1)[0]

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
            constr = self.cp.add_constraint_bath(self.Y[self.Nodes[i]] == ind)

            print("Branch: " + str(constr), "obj: " + str(obj))

            # Recursive call
            self.BnBMaxClique()

            # Remove constrain from the model
            self.cp.remove_constraint_bath(constr)

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
        check_time_iterations = 1000
        iteration_n = cycle(range(check_time_iterations + 1))
        while stack:

            if next(iteration_n) == check_time_iterations:
                if time.time() - self.start_time > self.timeout:
                    print("TimeOut!")
                    self.is_timeout = True
                    break

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
                    #print("Cut branch: ", self.cutted)
                    continue

                vals = sol.get_all_values()
                i = MaxCliqueProblem.get_node_index_to_branch(vals)

                # If it's integer solution
                if i == -1:
                    # If current solution better then previous
                    vals_to_set = [self.Nodes[i] for i, val in enumerate(vals) if val == 1.0]
                    if self.objective_best < obj:
                        print("--------------------Find solution: ", obj, '--------------------')
                        # Remember it
                        self.objective_best = obj
                        self.objective_best_vals = vals_to_set
                    else:
                        print("Another int solution")

                    print("Perform local search..")
                    self.local_search(vals_to_set)

                    continue

                # Set elem branch var
                i = self.Y[self.Nodes[i]]
                elem['constr'] = self.cp.add_constraint_bath(i == 0)

                #print("Branch by ", i, " obj: ", obj)
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
                self.cp.remove_constraint_bath(constr)
                elem["constr"] = self.cp.add_constraint_bath(constr.lhs == 1)

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
                self.cp.remove_constraint_bath(elem["constr"])

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

    for inp in Content_list:
        with open("contest.txt", "a") as outp:
            problem = MaxCliqueProblem(inp=inp)
            # Setup signal to two hours
            #signal.signal(signal.SIGALRM, handler)
            #signal.alarm(7200)

            # Set timeout on two hours
            problem.solve(timeout=7200)

            outp.write("Problem INP: " + str(problem.INP) + "\n")
            outp.write("Obj: " + str(problem.objective_best) + "\n")
            outp.write("Nodes: " + str(problem.objective_best_vals) + "\n")
            outp.write("Time elapsed second: " + str(problem.time_elapsed ) + "\n")
            outp.write("Time elapsed minutes: " + str(problem.time_elapsed / 60) + "\n")
            if problem.is_timeout:
                outp.write("Stopped by timer\n")
            outp.write("\n")
            print(problem.objective_best)
            print(problem.objective_best_vals)
            print("Time elapsed: ", problem.time_elapsed)