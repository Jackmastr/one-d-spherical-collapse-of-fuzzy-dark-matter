import pytest
import numpy as np
from collapse import SphericalCollapse

@pytest.fixture
def base_config():
    return {
        "G": 1,
        "N": 1,
        "r_max": 1,
        "r_min": 0,
        "m_tot": 1,
        "point_mass": 0,
        "j_coef": 1,
        "safety_factor": 1e-5,
        'dt_min': 1e-12,
        'H': 0,
        "stepper_strategy": "velocity_verlet",
        "energy_strategy": "kin_grav_rot",
        "timescale_strategy": "dyn",
        "thickness_strategy": "const",
        "t_max": 5,
        "save_dt": 1e-4,
    }

def generate_test_cases():
    m_values = [1]
    H_values = [0]
    r_values = [1]
    J_values = [1e-2]

    for m in m_values:
        for H in H_values:
            for r in r_values:
                for J in J_values:
                    yield (m, H, r, J)

@pytest.mark.parametrize("m,H,r,J", generate_test_cases())
def test_kepler_periapsis(base_config, m, H, r, J):
    # Define test parameters
    G = 1
    M = m
    # Calculate analytical values
    v = H * r
    L = J * m
    E_tot = (1/2)*m*v**2 - G*m*M/r + L**2/(2*m*r**2)
    a = -E_tot
    b = -G*M*m
    c = L**2/(2*m)
    r_close_analytical = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    r_close_analytical = np.nanmin([r_close_analytical, r])
    r_far_analytical = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    r_far_analytical = np.nanmax([r_far_analytical, r])

    # Set up simulation
    config = {**base_config, "m_tot": m, "j_coef": J, 'H': H, 'r_max': r}
    sim = SphericalCollapse(config)
    results = sim.run()

    # Extract simulation results
    sim_e_tot = results['e_tot']
    r_close_simulated = results['r'].min()
    r_far_simulated = results['r'].max()

    # Perform assertions
    tol = 1e-5
    for i, sim_e_tot_step in enumerate(sim_e_tot):
        if not pytest.approx(E_tot, abs=tol) == sim_e_tot_step.sum():
            e_r = results['e_r'][i]
            e_k = results['e_k'][i]
            e_g = results['e_g'][i]
            print(f"Expected total energy: {E_tot}, Got: {sim_e_tot_step.sum()}")
            print(f"r: {r}, e_r: {e_r}, e_k: {e_k}, e_g: {e_g}")
            assert pytest.approx(E_tot, abs=tol) == sim_e_tot_step.sum(), f"Total energy mismatch at time step {i}"
    
    assert pytest.approx(r_close_simulated, abs=tol) == r_close_analytical, "Periapsis distance mismatch"
    assert pytest.approx(r_far_simulated, abs=tol) == r_far_analytical, "Apoapsis distance mismatch"