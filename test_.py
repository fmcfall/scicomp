import pytest
import numpy as np
from all_ode import HopfBifurcation, Cubic
from shooting import shooting, get_ode_data
from continuation import pseudo_continuation, natural_continuation

class Test():
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
    
    def test_pseudo_sols_pars_len(self):
        ode = Cubic.func
        u0, par0 = Cubic.params()
        limit_cycle = False
        vary_par = 0
        step = 0.05
        max_steps = 80
        sols, pars = pseudo_continuation(ode, u0, limit_cycle, par0, vary_par, step, max_steps)
        assert len(sols) == len(pars)
        
    def test_natural_sols_pars_len(self):
        ode = Cubic.func
        u0, par0 = Cubic.params()
        limit_cycle = False
        vary_par = 0
        step = 0.05
        max_steps = 330
        sols, pars = natural_continuation(ode, u0, limit_cycle, par0, vary_par, step, max_steps)
        assert len(sols) == len(pars)