from docplex.mp.model import Model
import unittest

class TestPriorityQueue(unittest.TestCase):

    def test_constraints_add_remove(self):
        mod = Model(name='Test')

        v = [mod.continuous_var(name= 'y_{0}'.format(i)) for i in [1,2,3] ]
        i = 1
        constr = mod.add_constraint(v[0] == i)

        self.assertEqual(mod.number_of_linear_constraints, 1)
        mod.remove_constraint(constr)
        self.assertEqual(mod.number_of_linear_constraints, 0)
        


if __name__ == "__main__":
    unittest.main()