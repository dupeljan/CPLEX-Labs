# COPYRIGHT
# Daniil Lyakhov
# dupeljan@gmail.com
import numpy as np
from priorityQueue import PriorityQueue
from batchedModel import BatchedModel

import threading

from numba import jit

from functools import reduce
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
        #'c-fat200-1.clq',
        #'c-fat200-2.clq',
        #'c-fat200-5.clq',
        #'c-fat500-1.clq',
       # 'c-fat500-10.clq',
       # 'c-fat500-2.clq',
       # 'c-fat500-5.clq',
        #'MANN_a9.clq',
        #'hamming6-2.clq',
        #'hamming6-4.clq',
        'C125.9.clq',
        'gen200_p0.9_44.clq',
        'gen200_p0.9_55.clq',
        'san200_0.7_1.clq',
        'san200_0.7_2.clq',
        'san200_0.9_1.clq',
        'san200_0.9_2.clq',
        'san200_0.9_3.clq',
        'sanr200_0.7.clq',
       'keller4.clq',
        'brock200_1.clq',
        'brock200_3.clq',
        'brock200_4.clq',
        'p_hat300-1.clq',
       'p_hat300-2.clq',
	'brock200_2.clq',
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

    def _is_state_set(self, ind_set):
        """Test if ind_set is actually state set"""
        for i, j in comb(ind_set, 2):
            if self.G.has_edge(i, j):
                return False
        return True

    @jit(forceobj=True)
    def init_heuristic_static_set(self):
        """Trying to find best static set"""
        vertex_set = set(self.G.nodes)
        res = set()
        # Estimation of difference with true results
        est = 0
        # Auxiliary values
        # k(v) - neighbors count for vertex v
        # m(v) - count of edges need to
        # make subgraph consist of neighbors of v
        # full connected
        while vertex_set:
            # Init k and m
            k = 0
            m = self.G.size()
            # Find v_res : k(v_res) = max
            # and among all such vertexes m(v_res) = min
            # Init v_res
            v_res = -1
            for v in vertex_set:
                neighbors = set(self.G.neighbors(v)) & vertex_set
                k_upd = len(neighbors)
                m_upd = k_upd * (k_upd - 1) / 2 - nx.subgraph(self.G, neighbors).size()
                if m_upd < m or (m_upd == m and k_upd > k):
                    k = k_upd
                    m = m_upd
                    v_res = v

            assert v_res != -1, "Error, insert undefined vector"
            # Add new vertex to state set
            v_res = v_res
            res |= {v_res}
            # Remove all neighbors of v_res
            # from vertex_set
            vertex_set -= set(self.G.neighbors(v_res)) | {v_res}
            # Update estimation
            est += m

        # Return founded state set
        return res

    @jit
    def init_heuristic_clique(self):
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
                print("Perform local search..")
                self.local_clique_search()

    @jit
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

    #@jit
    def local_state_set_search(self, state_set_inp=None, weights=None):
        """Try to find better solution
        based on given"""

        class StateSet:
            """Incapsulate state set manipulations"""
            def __init__(self, state_set, G):
                # tau[n] =
                # edge count from state_set neighbor n
                # to state_set
                self.G = G
                self.tau = {n: 0 for n in self.G.nodes}
                self.state_set = set(state_set)
                self.state_set_list = []
                self.neighbors = reduce(lambda x, y: x | y, [set(self.G.neighbors(x)) for x in self.state_set]) - \
                                 self.state_set
                self._initzialized = False
                self.maximize_init()
                self._initzialized = True
                self.iter_index = 0


            def delete(self, value):
                """Delete value from the
                state set and update neighbors"""
                self.state_set -= {value}
                self.tau[value] = len(self.state_set) - 1
                # Delete neighbors which is connected
                # only with value
                neighbors = list(self.G.neighbors(value))
                neighbors_value_only = {n for n in neighbors if self.tau[n] == 1}
                self.neighbors -= set(neighbors_value_only)
                # Update tau if already initialized
                if self._initzialized:
                    for n in neighbors:
                        self.tau[n] -= 1
                        # if tau is equal to zero
                        # then we need to add this vertex to state set
                        if not self.tau[n]:
                            self.add({n})

                # rewind iterations
                self._rewind()

            def add(self, values):
                """Add values to state_set
                and update neighbors and tau.
                    values must be a set!"""
                # Add values to list
                self.state_set |= values
                # Update neighbors
                new_neighbors = reduce(lambda x, y: x | y, [set(self.G.neighbors(x)) for x in values])
                self.neighbors |= new_neighbors
                # Remove clique nodes from neighbors
                self.neighbors -= self.state_set
                # Update tau if already initialized
                if self._initzialized:
                    for n in new_neighbors:
                        self.tau[n] += 1


                # rewind iterations
                self._rewind()

            def maximize_init(self):
                """Try to maximize current state set
                just find appropriate node in the
                list of the neighbors"""
                repeat = True
                while repeat:
                    repeat = False

                    for n in self.neighbors:
                        b = len(set(self.G.neighbors(n)) & self.state_set)
                        self.tau[n] = b
                        # If we can add n to state set
                        if not b:
                            # add it and repeat from the begining
                            self.add({n})
                            repeat = True
                            break

            def _rewind(self):
                """Start iteration from the
                begining"""
                self.state_set_list = list(self.state_set)
                self.iter_index = 0

            def __iter__(self):
                self.state_set_list = list(self.state_set)
                return self

            def __next__(self):
                try:
                    self.iter_index += 1
                    return self.state_set_list[self.iter_index - 1]
                except IndexError:
                    raise StopIteration

            def __len__(self):
                return len(self.state_set)

        state_set = StateSet(set(state_set_inp), self.G)
        assert self._is_state_set(state_set.state_set), "ERROR"

        # Check is weighted ones
        weighted = weights is not None
        if weighted:
            weights_map = {self.Nodes[n]: w for n, w in enumerate(weights)}

        for x in state_set:

            # Find candidates to add
            candidates = {n for n in set(self.G.neighbors(x))
                                                if state_set.tau[n] == 1}

            # Choose best pair to swap
            if weighted:
                best = 0

                for c in candidates:
                    pair = candidates - set(self.G.neighbors(c)) - {c}
                    if pair:
                        # 1:2 weighted swap
                        pair = list(pair)
                        maximal_pair = pair[np.argmax([weights_map[n] for n in pair])]
                        score = weights_map[c] + weights_map[maximal_pair]
                        if not best or best["score"] < score:
                            best = dict(elems={c, maximal_pair}, score=score)
                    '''
                    # 1:1 weighted swap
                    else:
                        
                        if not best or best["score"] < weights_map[c]:
                            best = dict(elems={c}, score=weights_map[c])
                    '''
                if not best:
                    continue
                if best["score"] > weights_map[x]:
                    # Swap it!
                    print("Improve solution by ", best["score"])

                    # Update state set
                    state_set.add(best["elems"])
                    state_set.delete(x)

                    assert self._is_state_set(state_set.state_set), "State set is corupted!"

            else:
                for c in candidates:
                    # TODO sort and use set instead of set
                    pair = candidates - set(self.G.neighbors(c)) - {c}
                    # If we can add 2 nodes instead of one
                    if pair:

                        swap = (x, c, pair.pop())
                        print("Improve solution by 1!")

                        # Update state set
                        state_set.add(set(swap[1:]))
                        state_set.delete(swap[0])

                        assert self._is_state_set(state_set.state_set), "State set is corupted!"
                        break

        return state_set.state_set

    #@jit
    def local_clique_search(self, clique_inp=None):
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

    #@jit
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

    def several_separation(self, weights, count= 3):
        '''Call separation several times
        hope to get better constraints
        params:
            weights - weights for independent set
            count - number of separation call '''
        res = set()
        for i in range(count):
            res |= {tuple(self.separation(weights))}
        return res

    def separation(self, weights):
        """Find max weighted independent set
        by some heuristics
        params:
            weights - weights for independent set"""
        # Setup weights as w / color
        colors = nx.algorithms.coloring.greedy_color(self.G, strategy='random_sequential')
        weights = [w / (colors[self.Nodes[i]] + 1) for i, w in enumerate(weights)]
        # Greedy search:
        res = set()
        score = 0
        # Sort weights
        #values_sorted_list = sorted([(self.Nodes[n], w) for n, w in enumerate(weights)], key=lambda x: -x[1])
        # Set consist of allowed values
        #allowed = set(self.Nodes)

        # Greedy filtering
        # Put everything in priority queue
        q = PriorityQueue()
        q.heappify([[-w, self.Nodes[n]] for n, w in enumerate(weights)])
        '''
        # Get random first variable
        weights = np.array(weights)
        first, score = self.Nodes[np.random.choice(np.argwhere(weights == np.max(weights)).flatten())], np.max(weights)
        q.remove_task(first)
        '''
        while True:
            '''
            # Take biggest element form queue
            if first:
                pop = (score, first)
                first = False
            else:
            '''
            pop = q.pop_task_and_priority()
            if pop is None:
                break

            # Add it to our solution
            res |= {pop[1]}
            score += pop[0]

            # Remove all neighbors from queue
            for n in self.G.neighbors(pop[1]):
                q.remove_task(n)
        '''        
        for n, val in values_sorted_list:
            # If element is allowed to add to stable set
            if n in allowed:
                # Add note to heuristic solution and
                res |= {n}
                allowed -= set(self.G.neighbors(n)) | {n}
                score += val
        '''

        # Trying to improve it by local search
        res = self.local_state_set_search(state_set_inp=res, weights=weights)
        return res

    def corrupted_edges(self, values):
        """Called if only all values is integer.
        return list of node pairs from
        the solution, which is no connected"""
        nodes = [self.Nodes[n] for n, val in enumerate(values) if val == 1]
        # Set of corrupt edges in clique
        corrupt = set()
        for i, j in comb(nodes, 2):
            if not self.G.has_edge(i, j):
                corrupt |= {(i, j)}
        '''        
        for i, n in enumerate(nodes):
            neighbors = list(self.G.neighbors(n))
            for j in nodes[i+1:]:
                if j not in neighbors:
                    corrupt |= {(n, j)}
        '''

        return corrupt


    def __init__(self, inp, mode="BNB"):
        self.INP = inp
        self.mode = mode
        self._conf = False
        self.get_input()
        # Create graph
        self.G = nx.Graph()
        self.G.add_edges_from(self.Edges)
        self.objective_best = 0
        self.objective_best_vals = 0
        print("Start conf model")
        if mode == "BNB":
            self.configure_model_BnB()
        elif mode == "BNC":
            self.configure_model_BnC()
        print("End conf model")
        self.cutted = 0
        self.time_elapsed = 0
        self.is_timeout = False
        self.check_time_iterations = 10000
        self.iteration_n = cycle(range(self.check_time_iterations + 1))

    def solve(self, timeout=7200):
        # CP must be CPLEX model
        # inicialized for MaxClique
        # problem
        self.timeout = timeout
        assert self._conf, "Configurate model first!"
        # Try to find heuristic solution
        self.init_heuristic_clique()
        assert self._is_best_vas_clique(), "ERROR"
        print("Start to solve")
        #self.BnBMaxClique()
        # Limit solving on time
        self.start_time = time.time()
        if self.mode == "BNB":
            self.BnBMaxCliqueDFS()
        elif self.mode == "BNC":
            self.BnCMaxClique()
        self.time_elapsed = time.time() - self.start_time

    def get_input(self):
        #self.INP = ['c125.9.txt', 'keller4.txt', 'p_hat300_1.txt', 'brock200_2.txt'][self.inp_no]
        self.Edges = [list(map(int, str_.split()[1:3])) for str_ in open('input/DIMACS_all_ascii/'+self.INP).readlines() if str_[0] == 'e']
        self.Nodes = list(set([y for x in self.Edges for y in x]))
        # Set variable to protect BnB metod from unconfigurate model
        self._conf = False

    def define_model_and_variables(self):
        self.cp = BatchedModel(name='Max_clique')
        # Continious model vars
        self.Y = {i: self.cp.continuous_var(name='y_{0}'.format(i)) for i in self.Nodes}
        # y constrains
        self.Y_cons = {n: self.cp.add_constraint_bath(self.Y[n] <= 1) for n in self.Nodes}
        # Set objective
        self.cp.maximize(self.cp.sum(self.Y))

    def add_coloring_constraints(self):
        # Add constrains: nodes cant be in one clique
        # if you can color them in one color

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
        for i in range(50):#300
            coloring_list.append(self.colors_to_indep_set(nx.algorithms.coloring.greedy_color(self.G,
                                                                                        strategy='random_sequential')))
        coloring_list = sorted(coloring_list, key=lambda x: len(x.keys()))

        for elem in coloring_list[:20]:#[:]
            ind_set_maximal = self.maximal_ind_set_colors(elem)
            comps |= {self.cp.sum([self.Y[i] for i in x]) <= 1 for x in ind_set_maximal.values()}

        # Add constraints
        self.cp.add_constraints(comps)

    def configure_model_BnC(self):
        self.define_model_and_variables()
        heurist_static_set = self.init_heuristic_static_set()
        assert self._is_state_set(heurist_static_set), "Error, heuristic return connected set"
        self.cp.add_constraint_bath(self.cp.sum([self.Y[n] for n in heurist_static_set]) <= 1)
        self.add_coloring_constraints()
        self._conf = True

    def configure_model_BnB(self):
        self.define_model_and_variables()
        # Constrains for clique
        # Natural constraints
        bath = []
        for i, j in comb(self.Nodes, 2):
            if [i, j] not in self.Edges and [j, i] not in self.Edges:
                bath += [self.Y[i] + self.Y[j] <= 1]

        self.cp.add_constraints(bath)

        self.add_coloring_constraints()

        # Allow BnB to work
        self._conf = True

        print("Constraints count: ", self.cp.number_of_constraints)

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

    def BnCMaxClique(self):
        """Compute optimal solutuon
         for max clique problem using
         Branch and Cuts method
         """

        # Check timeout
        if next(self.iteration_n) == self.check_time_iterations:
            if time.time() - self.start_time > self.timeout:
                print("TimeOut!")
                self.is_timeout = True
                self.iteration_n = cycle([self.check_time_iterations])
                return

        # Solve the problem
        # in current state
        sol = self.cp.solve()

        # If there is no solution
        if sol is None:
            #print("None solution found")
            # Cut it
            return

        obj = sol.get_objective_value()
        val = sol.get_all_values()

        # If current branch upper bound not more
        # than known objective
        if trunc_precisely(obj) <= self.objective_best:
            self.cutted += 1
            #print("Cut branch. Obj: ", obj, " Cutting count: ", self.cutted)
            # Cut it
            return

        # Cutting
        eps = 0.01
        obj_pred = obj
        cut_branch = False
        # How much iteration cycle should wait and add cutts
        iter_without_changing = 10
        i = 0
        while True:
            several_sep = self.several_separation(val, count=10)
            for sep in [s for s in several_sep if len(s) > 2]:
                self.cp.add_constraint_bath(self.cp.sum([self.Y[n] for n in sep]) <= 1)
            sol = self.cp.solve()

            if sol is None:
                cut_branch
                break

            obj_new = sol.get_objective_value()
            # If current branch worst than best solution- cut it
            if trunc_precisely(obj_new) < self.objective_best:
                cut_branch = True
                break

            # Stop iterations while difference between solutions is too little
            if obj_pred - obj_new < eps:
                # i += 1
                #if i == iter_without_changing:
                #    break
                break

            # Change obj pred before next iteration
            obj_pred = obj_new

        obj = obj_pred
        #print("Done cutting, obj: ", obj)

        if cut_branch:
            return

        # Get best branching value
        i = MaxCliqueProblem.get_node_index_to_branch(val)
        # If current solution better then previous
        if i == -1:
            corrupted = self.corrupted_edges(val)
            if corrupted:
                for c in list(corrupted)[:100]:
                    self.cp.add_constraint_bath(self.Y[c[0]] + self.Y[c[1]] <= 1)

                self.BnCMaxClique()

            elif obj > self.objective_best:
                print("--------------------Find solution: ", obj, '--------------------')
                # Remember it
                self.objective_best = obj
                self.objective_best_vals = {self.Nodes[i] for i, n in enumerate(val) if n == 1}
                # Perform local clique search
                self.local_clique_search()
            return

        # Else - branching

        for ind in [0, 1]:
            # Set i-th constraint to val
            constr = self.cp.add_constraint_bath(self.Y[self.Nodes[i]] == ind)

            #print("Branch: " + str(constr), "obj: " + str(obj))

            # Recursive call
            self.BnCMaxClique()

            # Remove constrain from the model
            self.cp.remove_constraint_bath(constr)

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
        if trunc_precisely(obj) <= self.objective_best:
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
        #cons = self.cp.number_of_constraints
        #cons_before = 0
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
                    self.local_clique_search(vals_to_set)

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
        with open("contest_bnc.txt", "a") as outp:
            print("Start to solve problem ", inp)
            problem = MaxCliqueProblem(inp=inp, mode="BNC")
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
