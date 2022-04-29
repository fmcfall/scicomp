import pytest
import numpy as np
from all_ode import HopfBifurcation
from shooting import shooting, get_ode_data

class TestShootingMethod():
    def test_output_length(self):
        args = np.array([0.5, -1])
        u0 = [1.5, 0, 20]
        sol = shooting(HopfBifurcation.func, u0, args=args)
        assert len(sol) == len(u0)

    def test_data_length(self):
        args = np.array([0.5, -1])
        u0 = [1.5, 0, 20]
        y, t = get_ode_data(HopfBifurcation.func, u0, args=args)
        for i in y:
            assert len(i) == len(t)
    