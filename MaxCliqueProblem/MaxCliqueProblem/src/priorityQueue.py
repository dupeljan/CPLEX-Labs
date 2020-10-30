from heapq import heappush, heappop, heapify
import numpy as np

class PriorityQueue:
    """Implemets fast realisation
    of priority queue in terms of 
    tasks
    """
    def __init__(self):
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of tasks to entries
        self.REMOVED = np.inf      # placeholder for a removed task

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        entry = [-priority, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def heappify(self, list_):
        """Heapify inp list
        params:
            list_: must contain two elements lists
                first elem of each list is minus priority,
                second elem of each list is task"""
        self.pq = list_
        self.entry_finder = {entity[1]: entity for entity in list_}
        heapify(self.pq)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        try:
            entry = self.entry_finder.pop(task)
            entry[-1] = self.REMOVED
        except KeyError:
            pass

    def pop_task_and_priority(self):
        'Remove and return the lowest priority task and the priority. None if empty.'
        while self.pq:
            # Needs fix here
            priority, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return -priority, task

        return None
