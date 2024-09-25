import pytest
import numpy as np
from collapse import SphericalCollapse
from scipy.integrate import solve_ivp
import os
import matplotlib.pyplot as plt

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
        "ang_mom_strategy": "const",
        "safety_factor": 1e-5,
        'dt_min': 1e-16,
        'H': 0,
        "stepper_strategy": "beeman",
        "energy_strategy": "kin_grav_rot",
        "timescale_strategy": "dyn_vel",
        "thickness_strategy": "const",
        "t_max": 200,
        "save_dt": 1e-5,
        "save_strategy": "vflip",
    }

def generate_test_cases():
    base_case = (1, 0, 1, 1e-1)
    m_values = [0.1, 10]
    H_values = [0.5, 1]
    r_values = [0.1, 10]
    J_values = [1e-3, 1e-2]

    for m in m_values:
        yield (m, *base_case[1:])
    for H in H_values:
        yield (*base_case[:1], H, *base_case[2:])
    for r in r_values:
        yield (*base_case[:2], r, *base_case[3:])
    for J in J_values:
        yield (*base_case[:3], J)

@pytest.mark.parametrize("m,H,r,J", generate_test_cases())
def test_kepler(base_config, m, H, r, J):
    """
    Test the simulation's r(t) against a numerically solved r(t) using scipy.
    """
    initial_conditions = [r, H * r]  # [r(0), v(0)]

    def acceleration(t, y):
        r, v = y
        a = -G * m / r**2 + (J ** 2) / r**3
        return [v, a]

    # Define test parameters
    G = 1
    M = m
    # Calculate analytical values
    v = H * r
    L = J * m
    E_k = (1/2)*m*v**2
    E_g = -G*m*M/r
    E_rot = L**2/(2*m*r**2)
    E_tot = E_k + E_g + E_rot
    a = -E_tot
    b = -G*M*m
    c = L**2/(2*m)
    r_close_analytical = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    r_close_analytical = np.nanmin([r_close_analytical, r])
    r_far_analytical = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    r_far_analytical = np.nanmax([r_far_analytical, r])
    t_max = np.sqrt(4*np.pi/m * r_far_analytical**3)
    print(t_max)

    # Set up simulation
    config = {**base_config, "m_tot": m, "j_coef": J, 'H': H, 'r_max': r, 't_max': t_max}
    sim = SphericalCollapse(config)
    results = sim.run()

    # Extract simulation results
    sim_e_tot = results['e_tot']
    r_close_simulated = results['r'].min()
    r_far_simulated = results['r'].max()

        # Time span for the simulation
    t_span = (0, t_max)
    t_eval = np.linspace(t_span[0], t_span[1], int(t_max / base_config["save_dt"]) + 1)

    # Solve ODE numerically
    sol = solve_ivp(acceleration, t_span, initial_conditions, t_eval=t_eval, method='DOP853', rtol=1e-12)

    if not sol.success:
        pytest.fail("ODE solver failed to integrate.")

    r_numerical = sol.y[0]
    v_numerical = sol.y[1]
    t_numerical = sol.t

    # Run simulation
    config = {**base_config, "m_tot": m, "j_coef": J, 'H': H, 'r_max': r, 't_max': t_max, 'save_strategy': 'vflip'}
    sim = SphericalCollapse(config)
    results = sim.run()

    r_simulated = results['r'].reshape(-1)
    t_simulated = results['t'].reshape(-1)

    # Interpolate numerical solution to simulation time points
    r_expected = np.interp(t_simulated, t_numerical, r_numerical)



    # Perform assertions
    tol = 1e-5
    # for i, sim_e_tot_step in enumerate(sim_e_tot):
    #     if not pytest.approx(E_tot, rel=tol) == sim_e_tot_step.sum():
    #         e_r = results['e_r'][i]
    #         e_k = results['e_k'][i]
    #         e_g = results['e_g'][i]
    #         print(f"Expected total energy: {E_tot}, Got: {sim_e_tot_step.sum()}")
    #         print(f"r: {r}, e_r: {e_r}, e_k: {e_k}, e_g: {e_g}")
    #         assert pytest.approx(E_tot, rel=tol) == sim_e_tot_step.sum(), f"Total energy mismatch at time step {i}"
    
    assert pytest.approx(r_close_analytical, rel=tol) == r_close_simulated, "Periapsis distance mismatch"
    assert pytest.approx(r_far_analytical, rel=tol) == r_far_simulated, "Apoapsis distance mismatch"


        # Define tolerance
    tol = 1e-1

    # Perform assertions
    try:
        np.testing.assert_allclose(r_simulated, r_expected, rtol=tol, err_msg="Simulated r(t) does not match numerical solution.")
    except AssertionError as e:
        # Create directory for plots if it doesn't exist
        plot_dir = "test_outputs/r_t_comparison_plots"
        os.makedirs(plot_dir, exist_ok=True)

        # Generate a unique filename based on test parameters
        plot_filename = f"r_t_comparison_m{m}_H{H}_r{r}_J{J}.png"
        plot_path = os.path.join(plot_dir, plot_filename)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(t_simulated, r_expected, label='Numerical ODE Solution', linestyle='--')
        plt.plot(t_simulated, r_simulated, label='Simulation', alpha=0.7)
        plt.xlabel('Time (t)')
        plt.ylabel('Radial Distance (r)')
        plt.title(f'r(t) Comparison\nm={m}, H={H}, r₀={r}, J={J}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        # Optionally, log the path to the plot
        pytest.fail(f"Simulated r(t) does not match numerical solution.\nPlot saved to {plot_path}")
    else:
        # Optionally, save plots even if test passes for verification
        plot_dir = "test_outputs/r_t_comparison_plots"
        os.makedirs(plot_dir, exist_ok=True)

        plot_filename = f"r_t_comparison_m{m}_H{H}_r{r}_J{J}.png"
        plot_path = os.path.join(plot_dir, plot_filename)

        plt.figure(figsize=(10, 6))
        plt.plot(t_simulated, r_expected, label='Numerical ODE Solution', linestyle='--')
        plt.plot(t_simulated, r_simulated, label='Simulation', alpha=0.7)
        plt.xlabel('Time (t)')
        plt.ylabel('Radial Distance (r)')
        plt.title(f'r(t) Comparison\nm={m}, H={H}, r₀={r}, J={J}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

# @pytest.mark.slow
# @pytest.mark.parametrize("m,H,r,J", generate_test_cases())
# def test_r_t_comparison(base_config, m, H, r, J):
#     """
#     Test the simulation's r(t) against a numerically solved r(t) using scipy.
#     """
#     initial_conditions = [r, H * r]  # [r(0), v(0)]

#     def acceleration(t, y):
#         r, v = y
#         a = -G * m / r**2 + (J ** 2) / r**3
#         return [v, a]
    
#     # Define test parameters
#     G = 1
#     M = m
#     # Calculate analytical values
#     v = H * r
#     L = J * m
#     E_k = (1/2)*m*v**2
#     E_g = -G*m*M/r
#     E_rot = L**2/(2*m*r**2)
#     E_tot = E_k + E_g + E_rot
#     a = -E_tot
#     b = -G*M*m
#     c = L**2/(2*m)
#     r_close_analytical = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
#     r_close_analytical = np.nanmin([r_close_analytical, r])
#     r_far_analytical = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
#     r_far_analytical = np.nanmax([r_far_analytical, r])
#     t_max = np.sqrt(4*np.pi/m * r_far_analytical**3)
#     print(t_max)

#     # Time span for the simulation
#     t_span = (0, t_max)
#     t_eval = np.linspace(t_span[0], t_span[1], int(t_max / base_config["save_dt"]) + 1)

#     # Solve ODE numerically
#     sol = solve_ivp(acceleration, t_span, initial_conditions, t_eval=t_eval, method='DOP853', rtol=1e-12)

#     if not sol.success:
#         pytest.fail("ODE solver failed to integrate.")

#     r_numerical = sol.y[0]
#     v_numerical = sol.y[1]
#     t_numerical = sol.t

#     # Run simulation
#     config = {**base_config, "m_tot": m, "j_coef": J, 'H': H, 'r_max': r, 't_max': t_max, 'save_strategy': 'vflip'}
#     sim = SphericalCollapse(config)
#     results = sim.run()

#     r_simulated = results['r'].reshape(-1)
#     t_simulated = results['t'].reshape(-1)

#     # Interpolate numerical solution to simulation time points
#     r_expected = np.interp(t_simulated, t_numerical, r_numerical)

#     # Define tolerance
#     tol = 1e-1

#     # Perform assertions
#     try:
#         np.testing.assert_allclose(r_simulated, r_expected, rtol=tol, err_msg="Simulated r(t) does not match numerical solution.")
#     except AssertionError as e:
#         # Create directory for plots if it doesn't exist
#         plot_dir = "test_outputs/r_t_comparison_plots"
#         os.makedirs(plot_dir, exist_ok=True)

#         # Generate a unique filename based on test parameters
#         plot_filename = f"r_t_comparison_m{m}_H{H}_r{r}_J{J}.png"
#         plot_path = os.path.join(plot_dir, plot_filename)

#         # Plotting
#         plt.figure(figsize=(10, 6))
#         plt.plot(t_simulated, r_expected, label='Numerical ODE Solution', linestyle='--')
#         plt.plot(t_simulated, r_simulated, label='Simulation', alpha=0.7)
#         plt.xlabel('Time (t)')
#         plt.ylabel('Radial Distance (r)')
#         plt.title(f'r(t) Comparison\nm={m}, H={H}, r₀={r}, J={J}')
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(plot_path)
#         plt.close()

#         # Optionally, log the path to the plot
#         pytest.fail(f"Simulated r(t) does not match numerical solution.\nPlot saved to {plot_path}")
#     else:
#         # Optionally, save plots even if test passes for verification
#         plot_dir = "test_outputs/r_t_comparison_plots"
#         os.makedirs(plot_dir, exist_ok=True)

#         plot_filename = f"r_t_comparison_m{m}_H{H}_r{r}_J{J}.png"
#         plot_path = os.path.join(plot_dir, plot_filename)

#         plt.figure(figsize=(10, 6))
#         plt.plot(t_simulated, r_expected, label='Numerical ODE Solution', linestyle='--')
#         plt.plot(t_simulated, r_simulated, label='Simulation', alpha=0.7)
#         plt.xlabel('Time (t)')
#         plt.ylabel('Radial Distance (r)')
#         plt.title(f'r(t) Comparison\nm={m}, H={H}, r₀={r}, J={J}')
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(plot_path)
#         plt.close()
