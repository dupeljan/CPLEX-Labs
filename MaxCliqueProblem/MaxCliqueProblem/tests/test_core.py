import unittest
import context
from src import  is_int_list

class TestPriorityQueue(unittest.TestCase):

    def test_constraints_add_remove(self):
        x = [1.000000000001, 2.999999999999999999999]
        y = [1.000000000001, 8.003, 2.999999999999999999999]

        self.assertEqual(is_int_list(x), True)
        self.assertEqual(is_int_list(y), False)

        


if __name__ == "__main__":
    unittest.main()