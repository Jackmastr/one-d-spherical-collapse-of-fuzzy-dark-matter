import unittest
import numpy as np
import matplotlib.pyplot as plt
from collapse import SphericalCollapse
from utils import load_simulation_data

class TestKepler(unittest.TestCase):
    def setUp(self):
        # Set up the Kepler problem configuration
        self.config = {
            "G": 1,
            "N": 1,
            "r_max": 1,
            "r_min": 0,
            "m_tot": 1,
            "point_mass": 0,
            "j_coef": 1,
            "safety": 1e-4,
            'H': 0,
            "stepper_strategy": "velocity_verlet",
            "density_strategy": "const",
            "ang_mom_strategy": "const",
            "soft_func_strategy": "const_soft",
            "accel_strategy": "soft_grav",
            "m_enc_strategy": "const_inclusive",
            "r_ta_strategy": "r_is_r_ta",
            "intial_v_strategy": "hubble",
            "energy_strategy": "kin_grav_rot",
            "timescale_strategy": "dyn",
            "timestep_strategy": "simple_adaptive",
            "thickness_strategy": "const",
            "t_max": 5,
            "save_dt": 1e-4,
            "output_file": "kepler_test.h5"
        }
    
# G = 1
# m = 1e-4
# M = 1
# J = 1e-1
# E_tot = -G*M*m + 0.5*m*J**2
# a = -E_tot
# b = -G*M*m
# c = 0.5*m*J**2
# r_close = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
# print(r_close)

    def test_kepler_periapsis(self):
        m = 100
        v = 0
        r = 1
        J = 1
        L = J/m
        G = 1
        M = 1
        E_tot = (1/2)*m*v**2 - G*M*m/r + L**2/(2*m*r**2)
        assert np.isclose(E_tot, E_tot) #TODO: set equal to the value from results
        a = -E_tot
        b = -G*M*m
        c = L**2/(2*m)
        r_close_analytical = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
        print(r_close_analytical)
        sim = SphericalCollapse(self.config)
        results = sim.run()
        r = results['r']
        r_close_simulated = np.min(r)
        print(r_close_analytical)
        print(r_close_simulated)
        self.assertAlmostEqual(r_close_simulated, r_close_analytical)

if __name__ == '__main__':
    unittest.main()