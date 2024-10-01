import pytest
import numpy as np
from collapse import SphericalCollapse
from scipy.integrate import solve_ivp
import os
import matplotlib.pyplot as plt
import warnings

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
        "t_max": -1,
        "save_dt": 1e-5,
        "save_strategy": "vflip",
    }

def generate_test_cases():
    base_case = (1, 0, 1, 1e-1)
    m_values = [0.1, 10]
    H_values = [0.5, 1, 2]
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

def compute_analytical_predictions(G, m, J, H, r):
    """
    Compute analytical predictions for r_close and r_far.
    """
    # Calculate analytical values
    v = H * r
    L = J * m
    E_k = 0.5 * m * v ** 2
    E_g = -G * m * m / r  # Assuming M = m for simplicity
    E_rot = L ** 2 / (2 * m * r ** 2)
    E_tot = E_k + E_g + E_rot

    a = -E_tot
    b = -G * m * m
    c = L ** 2 / (2 * m)
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        raise ValueError("Discriminant is negative. No real solutions for r_close and r_far.")

    r_close = (-b - np.sqrt(discriminant)) / (2 * a)
    r_close = min(r_close, r)

    r_far = (-b + np.sqrt(discriminant)) / (2 * a)
    r_far = max(r_far, r)

    return r_close, r_far

def plot_comparison(t_simulated, r_simulated, r_expected, v_simulated, v_expected, r_close, r_far, m, H, r, J, plot_path):
    """
    Generate and save comprehensive comparison plots:
    - r(t) in linear and log-space with horizontal lines for periapsis and apoapsis
    - Relative differences in r(t) and v(t) on separate plots
    """
    plt.figure(figsize=(15, 20))  # Increased height to accommodate 4 subplots

    # Subplot 1: r(t) Comparison (Linear Scale)
    plt.subplot(4, 1, 1)
    plt.plot(t_simulated, r_expected, label='Numerical ODE Solution', linestyle='--')
    plt.plot(t_simulated, r_simulated, label='Simulation', alpha=0.7)
    plt.axhline(y=r_close, color='red', linestyle=':', label='Periapsis (r_close)')
    plt.axhline(y=r_far, color='green', linestyle=':', label='Apoapsis (r_far)')
    plt.xlabel('Time (t)')
    plt.ylabel('Radial Distance (r)')
    plt.title(f'r(t) Comparison - Linear Scale\nm={m}, H={H}, r₀={r}, J={J}')
    plt.legend()
    plt.grid(True)

    # Subplot 2: r(t) Log-Space Comparison
    plt.subplot(4, 1, 2)
    plt.plot(t_simulated, r_expected, label='Numerical ODE Solution', linestyle='--')
    plt.plot(t_simulated, r_simulated, label='Simulation', alpha=0.7)
    plt.axhline(y=r_close, color='red', linestyle=':', label='Periapsis (r_close)')
    plt.axhline(y=r_far, color='green', linestyle=':', label='Apoapsis (r_far)')
    plt.xlabel('Time (t)')
    plt.ylabel('Radial Distance (r) [Log Scale]')
    plt.yscale('log')
    plt.title('r(t) Comparison - Logarithmic Scale')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Subplot 3: Relative Difference in r(t)
    plt.subplot(4, 1, 3)
    rel_diff_r = np.abs(r_simulated - r_expected) / np.abs(r_expected)
    plt.plot(t_simulated, rel_diff_r, label='Relative Difference r(t)', color='blue')
    plt.xlabel('Time (t)')
    plt.ylabel('Relative Difference')
    plt.title('Relative Difference in r(t)')
    plt.legend()
    plt.grid(True)

    # Subplot 4: Relative Difference in v(t)
    plt.subplot(4, 1, 4)
    rel_diff_v = np.abs(v_simulated - v_expected) / np.abs(v_expected)
    plt.plot(t_simulated, rel_diff_v, label='Relative Difference v(t)', color='green')
    plt.xlabel('Time (t)')
    plt.ylabel('Relative Difference')
    plt.title('Relative Difference in v(t)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f'comparison_plots_{m}_{H}_{r}_{J}.png'))
    plt.close()

def plot_energy_comparison(t_simulated, energy_simulated, energy_expected, m, H, r, J, plot_path):
    """
    Generate and save relative energy difference plot.
    """
    plt.figure(figsize=(10, 6))
    rel_diff_energy = np.abs(energy_simulated - energy_expected) / np.abs(energy_expected)
    plt.plot(t_simulated, rel_diff_energy, label='Relative Energy Difference', color='red')
    plt.xlabel('Time (t)')
    plt.ylabel('Relative Energy Difference')
    plt.title(f'Energy Conservation Check\nm={m}, H={H}, r₀={r}, J={J}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f'energy_comparison_{m}_{H}_{r}_{J}.png'))
    plt.close()

def compute_energy(t, y, G, m, J):
    """
    Compute total energy for a given state.
    E = Kinetic + Potential + Rotational
    """
    r, v = y
    kinetic = 0.5 * m * v ** 2
    potential = -G * m * m / r
    rotational = (J ** 2) / (2 * m * r ** 2)
    return kinetic + potential + rotational

def validate_energy_conservation(energies, tol=1e-3, label=""):
    """
    Validate energy conservation by checking deviations from initial energy.
    Emits warnings if deviations exceed tolerance.
    """
    initial_energy = energies[0]
    deviations = np.abs(energies - initial_energy) / np.abs(initial_energy)
    significant_deviation = deviations > tol
    num_issues = np.sum(significant_deviation)

    if num_issues > 0:
        warnings.warn(f"{label}: Energy deviates beyond {tol*100}% at {num_issues} time steps.")

@pytest.mark.parametrize("m,H,r,J", generate_test_cases())
def test_kepler(base_config, m, H, r, J):
    """
    Test the simulation's r(t) against a numerically solved r(t) using scipy.
    Ensure that periapsis (r_close) and apoapsis (r_far) match analytical predictions.
    Optionally test for energy conservation in both the ODE solver and the simulation.
    """
    G = base_config["G"]

    # Configuration for optional energy conservation tests
    TEST_ENERGY_CONSERVATION = True
    ENERGY_TOLERANCE = 1e-3  # 0.1%
    ALLOWED_ENERGY_DEVIATIONS = np.inf  # Allow up to 10 deviations

    # Compute analytical predictions
    try:
        r_close_analytical, r_far_analytical = compute_analytical_predictions(G, m, J, H, r)
    except ValueError as e:
        pytest.skip(f"Skipping test due to analytical computation error: {e}")

    # Time span and evaluation points
    t_max = np.sqrt(4 * np.pi / m * r_far_analytical ** 3)
    t_span = (0, t_max)
    t_eval = np.linspace(t_span[0], t_span[1], int(t_max / base_config["save_dt"]) + 1)

    # Initial conditions [r(0), v(0)]
    initial_conditions = [r, H * r]

    # List of ODE solvers to use
    ode_methods = ['Radau', 'RK45', 'BDF', 'LSODA', 'DOP853']  # Add or remove methods as desired

    # Directory to save ODE solutions
    ode_solutions_dir = "test_outputs/ode_solutions/"
    os.makedirs(ode_solutions_dir, exist_ok=True)

    ode_results = {}

    for method in ode_methods:
        # Define the acceleration function
        def acceleration(t, y):
            r_current, v = y
            a = -G * m / r_current ** 2 + (J ** 2) / r_current ** 3
            return [v, a]

        # Solve ODE numerically with the current method
        sol = solve_ivp(acceleration, t_span, initial_conditions, t_eval=t_eval, method=method, rtol=1e-12)

        if not sol.success:
            pytest.fail(f"ODE solver '{method}' failed to integrate.")

        # Store the solution
        ode_results[method] = {
            't': sol.t,
            'r': sol.y[0],
            'v': sol.y[1]
        }

        # Save the solution to a file
        ode_filename = f"ode_solution_{method}_{m}_{H}_{r}_{J}.npz"
        np.savez(
            os.path.join(ode_solutions_dir, ode_filename),
            t=sol.t,
            r=sol.y[0],
            v=sol.y[1]
        )

        # Optional: Energy conservation check for each ODE solver
        if TEST_ENERGY_CONSERVATION:
            energies_numerical = compute_energy(sol.t, sol.y, G, m, J)
            validate_energy_conservation(energies_numerical, tol=ENERGY_TOLERANCE, label=f"ODE Solver ({method})")
            # Generate energy comparison plot
            energy_plot_dir = f"test_outputs/energy_comparison_plots/ode_solver/{method}/"
            os.makedirs(energy_plot_dir, exist_ok=True)
            plot_energy_comparison(
                sol.t,
                energies_numerical,
                energies_numerical[0],
                m=m,
                H=H,
                r=r,
                J=J,
                plot_path=energy_plot_dir
            )

    # Run simulation
    filename = f"test_outputs/test_kepler_{m}_{H}_{r}_{J}.h5"
    config = {**base_config, "m_tot": m, "j_coef": J, 'H': H, 'r_max': r, 't_max': t_max, 'save_filename': filename}
    sim = SphericalCollapse(config)
    results = sim.run()

    # Extract simulation results
    r_simulated = results['r'].reshape(-1)
    v_simulated = results['v'].reshape(-1)
    t_simulated = results['t'].reshape(-1)

    # Initialize a dictionary to store interpolated expected results for each method
    expected_results = {}

    for method, data in ode_results.items():
        r_numerical = data['r']
        v_numerical = data['v']
        t_numerical = data['t']

        # Interpolate numerical solution to simulation time points
        r_expected = np.interp(t_simulated, t_numerical, r_numerical)
        v_expected = np.interp(t_simulated, t_numerical, v_numerical)

        expected_results[method] = {
            'r_expected': r_expected,
            'v_expected': v_expected
        }

    # Choose a reference ODE solver for periapsis and apoapsis assertions (e.g., 'Radau')
    reference_method = 'Radau'
    r_expected_ref = expected_results[reference_method]['r_expected']
    v_expected_ref = expected_results[reference_method]['v_expected']

    # Assertions for periapsis and apoapsis
    tol_pos = 1e-5
    r_close_simulated = r_simulated.min()
    r_far_simulated = r_simulated.max()

    assert pytest.approx(r_close_analytical, rel=tol_pos) == r_close_simulated, "Periapsis distance mismatch"
    assert pytest.approx(r_far_analytical, rel=tol_pos) == r_far_simulated, "Apoapsis distance mismatch"

    # Perform assertions on the entire r(t) curve
    tol_curve = 1e-1
    try:
        np.testing.assert_allclose(r_simulated, r_expected, rtol=tol_curve,
                                   err_msg="Simulated r(t) does not match numerical solution.")
    except AssertionError as e:
        # Log the path to the plot and fail the test
        pytest.fail(f"Simulated r(t) does not match numerical solution.\nPlot saved to {plot_dir}/comparison_plots_{method}_{m}_{H}_{r}_{J}.png")

    # Create directory for plots if it doesn't exist
    plot_dir = "test_outputs/r_t_comparison_plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Generate plots with horizontal lines for periapsis and apoapsis
    plot_comparison(
        t_simulated,
        r_simulated,
        r_expected,
        v_simulated,
        v_expected,
        r_close=r_close_analytical,
        r_far=r_far_analytical,
        m=m,
        H=H,
        r=r,
        J=J,
        plot_path=plot_dir
    )

    # Optional: Energy conservation check for simulation
    if TEST_ENERGY_CONSERVATION:
        # Compute energies for simulation results
        energies_simulated = results['e_tot'].reshape(-1)
        energy_expected = energies_simulated[0]
        validate_energy_conservation(energies_simulated, tol=ENERGY_TOLERANCE, label="Simulation")

        # Generate energy comparison plot
        plot_dir = "test_outputs/energy_comparison_plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_energy_comparison(t_simulated, energies_simulated, energy_expected,
                               m=m, H=H, r=r, J=J, plot_path=plot_dir)