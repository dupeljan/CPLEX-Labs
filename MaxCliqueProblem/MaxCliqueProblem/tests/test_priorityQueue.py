import unittest

import os.path as osp
import sys
import inspect
current_dir = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = osp.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src import  PriorityQue


class TestPriorityQueue(unittest.TestCase):

    def test_order(self):
        pq = PriorityQueue()
        pq.add_task((3,3,3,3), 3)
        pq.add_task((2,2), 2)
        pq.add_task((1010,1010), 10)

        # Test order
        for x in [ (2, (2,2)), ( 3,(3,3,3,3)),( 10, (1010,1010))]:
            self.assertEqual(pq.pop_task_and_priority(),x)

if __name__ == "__main__":
    unittest.main()