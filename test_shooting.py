import unittest
import numpy as np
from all_ode import hopf_bifurcation
from shooting_method import shooting

class TestShootingMethod(unittest.TestCase):
    def test_output_length(self):
        args = np.array([0.5, -1])
        u0 = [1.5, 0, 20]
        sol = shooting(hopf_bifurcation, u0, args=args)
        self.assertEqual(len(sol),len(u0))

if __name__ == '__main__':
    unittest.main()

    