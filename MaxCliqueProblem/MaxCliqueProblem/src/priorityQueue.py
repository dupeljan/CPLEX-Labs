from heapq import heappush, heappop

class PriorityQueue:
    """Implemets fast realisation
    of priority queue in terms of 
    tasks
    """
    def __init__(self):
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of tasks to entries
        self.REMOVED = '<removed-task>'      # placeholder for a removed task

    def add_task(self,task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        entry = [-priority, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self,task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task_and_priority(self):
        'Remove and return the lowest priority task and the priority. Raise KeyError if empty.'
        while self.pq:
            priority, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return -priority, task
        raise KeyError('pop from an empty priority queue')
        
    def get_list_copy(self):
        'Return copy of all task list'
        return self.pq.copy()
    
    def __nonzero__(self):
        'Return value in conditional statements'
        return not not self.pq 
