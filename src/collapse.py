import numpy as np
from functools import partial
import h5py
import re
import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)

from simulation_strategies import *


class SphericalCollapse:
    def __init__(self, config=None):
        # Default parameters
        self.G = 1
        self.N = 100
        self.r_max = 1
        self.r_min = 1e-6
        self.m_tot = 1
        self.dt = 1e-5
        self.dt_min = 1e-9
        self.min_time_scale = None
        self.save_dt = 1e-4
        self.t_max = 2
        self.t = 0
        self.stepper = partial(Steppers.velocity_verlet, self)
        self.rho_func = partial(const_rho_func, self)
        self.j_func = partial(gmr_j_func, self)
        self.soft_func = partial(SofteningFunctions.const_soft_func, self)
        self.a_func = partial(AccelerationFunctions.soft_grav_a_func, self)
        self.m_enc_func = partial(EnclosedMassFunctions.m_enc_inclusive, self)
        self.r_ta_func = partial(r_is_r_ta_func, self)
        self.intial_v_func = partial(hubble_v_func, self)
        self.energy_func = partial(EnergyFunctions.default_energy_func, self)
        self.shell_vol_func = partial(keep_edges_shell_vol_func, self)
        self.timescale_func = partial(TimeScaleFunctions.rubin_loeb_cross_timescale_func, self)
        self.timestep_func = partial(TimeStepFunctions.const_timestep, self)
        self.thickness_func = partial(ShellThicknessFunction.const_shell_thickness, self)
        self.prev_r = None
        self.num_crossing = 0
        self.r = None
        self.v = None
        self.a = None
        self.a_prev = None
        self.m = None
        self.m_enc = None
        self.j = None
        self.j_coef = 1e-1
        self.thickness_coef = 0
        self.thicknesses = None
        self.softlen = 0
        self.e_tot = None
        self.e_g = None
        self.e_k = None
        self.e_r = None
        self.r_ta = None
        self.t_ta = None
        self.t_dyn = None
        self.t_vel = None
        self.t_acc = None
        self.t_cross = None
        self.t_zero = None
        self.t_rmin = None
        self.gamma = 0
        self.H = 0
        self.safety_factor = 1e-3
        self.shell_thickness = None
        self.snapshots = []
        self.save_to_file = False
        # Update if config dictionary is provided
        if config:
            for key, value in config.items():
                if callable(value) and not isinstance(value, partial):
                    setattr(self, key, partial(value, self))
                else:
                    setattr(self, key, value)
                    
        self.setup()

    def handle_reflections(self):
        inside_sphere = self.r < self.r_min
        self.r[inside_sphere] = 2 * self.r_min - self.r[inside_sphere]
        self.v[inside_sphere] = -self.v[inside_sphere]  # Reverse velocity for elastic collision


    def detect_shell_crossings(self):
        self.num_crossing = 0
        # Get the current radial positions
        current_r = self.r
        
        # If this is the first step, store the current positions and return 0
        if self.prev_r is None:
            self.prev_r = current_r.copy()
            return 0
        
        # Get the sorting indices for both previous and current positions
        prev_order = np.argsort(self.prev_r)
        current_order = np.argsort(current_r)
        
        # Count the number of shells in different order
        num_different_order = np.sum(prev_order != current_order)
        
        # Update prev_r for the next step
        self.prev_r = current_r.copy()
        
        self.num_crossing = num_different_order

    def setup(self):
        # Initialize radial positions
        self.r = np.linspace(self.r_max/self.N, self.r_max, self.N)
        # Initialize masses for each shell
        shell_volumes = self.shell_vol_func()
        densities = self.rho_func()
        self.m = shell_volumes * densities
        # Calculate initial enclosed mass
        self.m_enc = self.m_enc_func()
        # Initialize velocities
        self.v = self.intial_v_func()
        # Calculate initial turnaround radius
        self.r_ta = self.r_ta_func()
        # Initialize angular momentum
        self.j = self.j_func()
        # Calculate initial acceleration
        self.a = self.a_func()
        # Calculate initial energy
        self.energy_func()
        # timescales
        self.timescale_func()
        self.timestep_func()
        self.save()
        logger.info("Simulation setup complete")

    def run(self):
        next_save_time = self.save_dt
        next_progress_time = 0.1 * self.t_max  # Update progress every 10% of total time
        progress_interval = 0.1 * self.t_max
        while self.t < self.t_max:
            # Call the stepper to update positions, velocities, and accelerations
            self.stepper()
            self.timescale_func()
            self.timestep_func()
            self.detect_shell_crossings()
            # Save data if necessary
            if self.t >= next_save_time:
                # Update current energies
                self.energy_func()
                self.save()
                next_save_time = self.t + self.save_dt

            # Update progress
            if self.t >= next_progress_time:
                progress = (self.t / self.t_max) * 100
                print(f"Progress: {progress:.1f}%")
                next_progress_time += progress_interval

        if self.save_to_file:
            self.save_to_hdf5()
        return self.get_results_dict()

    def save(self):
        # Save relevant parameters of the simulation
        data = {
            't': self.t,
            'dt': self.dt,
            'r': self.r.copy(),
            'v': self.v.copy(),
            'a': self.a.copy(),
            'm_enc': self.m_enc.copy(),
            'e_tot': self.e_tot.copy(),
            'e_g': self.e_g.copy(),
            'e_k': self.e_k.copy(),
            'e_r': self.e_r.copy(),
            't_dyn': self.t_dyn,
            't_vel': self.t_vel,
            't_acc': self.t_acc,
            't_cross': self.t_cross,
            't_zero': self.t_zero,
            't_rmin': self.t_rmin,
            'num_crossing': self.num_crossing,
        }
        self.snapshots.append(data)

    def get_results_dict(self):
        results = {key: [] for key in self.snapshots[0].keys()}
        for snapshot in self.snapshots:
            for key, value in snapshot.items():
                results[key].append(value)
        return {key: np.array(value) for key, value in results.items()}  

    def save_to_hdf5(self, filename='simulation_data.h5'):
        """
        Save all snapshots to an HDF5 file, organized by parameter.
        """
        with h5py.File(filename, 'w') as hf:
            # Create datasets for each parameter
            num_snapshots = len(self.snapshots)
            num_shells = len(self.snapshots[0]['r'])
            
            # Create datasets
            t_dset = hf.create_dataset('t', (num_snapshots,), dtype='float64')
            dt_dset = hf.create_dataset('dt', (num_snapshots,), dtype='float64')
            r_dset = hf.create_dataset('r', (num_snapshots, num_shells), dtype='float64')
            v_dset = hf.create_dataset('v', (num_snapshots, num_shells), dtype='float64')
            a_dset = hf.create_dataset('a', (num_snapshots, num_shells), dtype='float64')
            m_enc_dset = hf.create_dataset('m_enc', (num_snapshots, num_shells), dtype='float64')
            e_tot_dset = hf.create_dataset('e_tot', (num_snapshots, num_shells), dtype='float64')
            e_g_dset = hf.create_dataset('e_g', (num_snapshots, num_shells), dtype='float64')
            e_k_dset = hf.create_dataset('e_k', (num_snapshots, num_shells), dtype='float64')
            e_r_dset = hf.create_dataset('e_r', (num_snapshots, num_shells), dtype='float64')

            # Fill datasets
            for i, snapshot in enumerate(self.snapshots):
                t_dset[i] = snapshot['t']
                dt_dset[i] = snapshot['dt']
                r_dset[i] = snapshot['r']
                v_dset[i] = snapshot['v']
                a_dset[i] = snapshot['a']
                m_enc_dset[i] = snapshot['m_enc']
                e_tot_dset[i] = snapshot['e_tot']
                e_g_dset[i] = snapshot['e_g']
                e_k_dset[i] = snapshot['e_k']
                e_r_dset[i] = snapshot['e_r']

        logger.info(f"Saved {num_snapshots} snapshots to {filename}")


    def __str__(self):
        def extract_function_name(func_repr):
            match = re.search(r"<function (\w+) at", func_repr)
            if match:
                return match.group(1)
            else:
                return "Unknown function"
        result = []
        for attr_name, attr_value in self.__dict__.items():
            if callable(attr_value):
                # If it's callable, print the attribute name
                result.append(f"{attr_name}: {extract_function_name(str(attr_value))}")
            else:
                # Otherwise, print the attribute name and its value
                result.append(f"{attr_name}: {attr_value}")
        return "\n".join(result)