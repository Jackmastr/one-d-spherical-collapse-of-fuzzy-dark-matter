import numpy as np
from functools import partial
import h5py
import logging
import types
from numba import jit, njit
from simulation_strategies import *
from utils import *
# Setup logging only once at the module level
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False  # Add this line

class SphericalCollapse:
    def __init__(self, config=None):
        # Default parameters
        self._set_default_parameters()
        
        # Update if config dictionary is provided
        if config:
            self._update_from_config(config)

        # Capture initial parameters
        self._initial_params = self._capture_initial_params()
                    
        self.setup()

    def _set_default_parameters(self):
        # Move all default parameter initialization here
        self.G = 1
        self.N = 100
        self.r_max = 1
        self.r_min = 1e-6
        self.m_tot = 1
        self.rho_bar = 0
        self.dt = 1e-5
        self.dt_min = 1e-9
        self.min_time_scale = None
        self.save_dt = 1e-4
        self.t_max = 2
        self.t = 0
        self.point_mass = 0
        self.stepper_strategy = "velocity_verlet"
        self.density_strategy = "const"
        self.ang_mom_strategy = "gmr"
        self.soft_func_strategy = "const_soft"
        self.accel_strategy = "soft_grav"
        self.m_enc_strategy = "overlap_inclusive"
        self.r_ta_strategy = "r_is_r_ta"
        self.intial_v_strategy = "hubble"
        self.energy_strategy = "kin_grav_rot"
        self.shell_vol_func = types.MethodType(keep_edges_shell_vol_func, self)
        self.timescale_strategy = "dyn_rmin"
        self.timestep_strategy = "simple_adaptive"
        self.thickness_strategy = "const"
        self.prev_r = None
        self.which_reflected = None
        self.refletion_events = []
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
        self.t_rmina = None
        self.gamma = 0
        self.H = 0
        self.safety_factor = 1e-3
        self.shell_thickness = None
        self.snapshots = []
        self.save_filename = None

    def _update_from_config(self, config):
        for key, value in config.items():
            if key not in self.__dict__:
                raise AttributeError(f"Attribute {key} does not exist in the object.")
            if callable(value):
                setattr(self, key, types.MethodType(value, self))
            else:
                setattr(self, key, value)

    def _capture_initial_params(self):
            """
            Capture all non-None and non-empty parameters before setup.
            """
            params = {}
            for attr, value in self.__dict__.items():
                if not attr.startswith('_') and value is not None and value != []:
                    if isinstance(value, (int, float, str, bool, np.number, np.ndarray)):
                        params[attr] = value
                    elif isinstance(value, types.MethodType):
                        # For methods, store the strategy name
                        params[attr] = value.__func__.__name__
            return params

    def get_parameters_dict(self):
        """
        Return the dictionary of initial simulation parameters.
        """
        return self._initial_params.copy()



    def handle_reflections(self):
        self.r, self.v, self.which_reflected = self._handle_reflections_numba(self.r, self.v, self.r_min)

    @staticmethod
    @njit
    def _handle_reflections_numba(r, v, r_min):
        which_reflected = np.zeros_like(r, dtype=np.bool_)
        for i in range(len(r)):
            if r[i] < r_min:
                r[i] = 2 * r_min - r[i]
                v[i] = -v[i]
                which_reflected[i] = True
        return r, v, which_reflected

    def detect_shell_crossings(self):
        pass

    def _initialize_strategies(self):
        strategy_mappings = {
            "stepper": (StepperFactory, self.stepper_strategy),
            "r_ta_func": (RTurnaroundFactory, self.r_ta_strategy),
            "a_func": (AccelerationFactory, self.accel_strategy),
            "soft_func": (SofteningFactory, self.soft_func_strategy),
            "m_enc_func": (EnclosedMassFactory, self.m_enc_strategy),
            "timescale_func": (TimeScaleFactory, self.timescale_strategy),
            "energy_func": (EnergyFactory, self.energy_strategy),
            "timestep_func": (TimeStepFactory, self.timestep_strategy),
            "thickness_func": (ShellThicknessFactory, self.thickness_strategy),
            "rho_func": (DensityFactory, self.density_strategy),
            "initial_v_func": (InitialVelocityFactory, self.intial_v_strategy),
            "j_func": (AngularMomentumFactory, self.ang_mom_strategy),
        }
        for attr_name, (factory, strategy_name) in strategy_mappings.items():
            try:
                strategy_instance = factory.create(strategy_name)
                if strategy_instance is None:
                    raise ValueError(f"Strategy creation for {attr_name} returned None")
                setattr(self, attr_name, types.MethodType(strategy_instance, self))
            except Exception as e:
                logger.error(f"Error initializing {attr_name}: {str(e)}")
                raise

    def setup(self):
        # Initialize factories
        self._initialize_strategies()
        # Initialize radial positions
        self.r = np.linspace(self.r_max/self.N, self.r_max, self.N)
        self.which_reflected = np.zeros_like(self.r, dtype=np.int32)
        # Initialize masses for each shell
        shell_volumes = self.shell_vol_func()
        densities = self.rho_func()
        self.m = shell_volumes * densities
        # Calculate initial enclosed mass
        self.m_enc = self.m_enc_func()
        # Initialize velocities
        self.v = self.initial_v_func()
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
        next_progress_time = 0.1 * self.t_max
        
        while self.t < self.t_max:
            self._update_simulation()
            next_save_time = self._save_if_necessary(next_save_time)
            next_progress_time = self._update_progress(next_progress_time)

        if self.save_filename:
            save_to_hdf5(self, self.save_filename)
        return self.get_results_dict()

    def _update_simulation(self):
        self.stepper()
        self.timescale_func()
        self.timestep_func()
        self.detect_shell_crossings()

    def _save_if_necessary(self, next_save_time):
        if self.t >= next_save_time:
            self.save()
            return self.t + self.save_dt
        return next_save_time

    def _update_progress(self, next_progress_time):
        if self.t >= next_progress_time:
            progress = int((self.t / self.t_max) * 100)
            print(f"Progress: {progress}%")
            return next_progress_time + 0.1 * self.t_max
        return next_progress_time

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
            't_rmina': self.t_rmina,
            'num_crossing': self.num_crossing,
        }
        self.snapshots.append(data)

    def get_results_dict(self):
        results = {key: [] for key in self.snapshots[0].keys()}
        for snapshot in self.snapshots:
            for key, value in snapshot.items():
                results[key].append(value)
        return {key: np.array(value) for key, value in results.items()}  
    
    def _capture_initial_params(self):
        """
        Capture all non-None and non-empty parameters before setup.
        """
        params = {}
        for attr, value in self.__dict__.items():
            if not attr.startswith('_') and value is not None and value != []:
                if isinstance(value, (int, float, str, bool, np.number, np.ndarray)):
                    params[attr] = value
                elif isinstance(value, types.MethodType):
                    # For methods, store the strategy name
                    params[attr] = value.__func__.__name__
        return params

    def get_parameters_dict(self):
        """
        Return the dictionary of initial simulation parameters.
        """
        return self._initial_params.copy()


    def __str__(self):
        result = []
        for attr_name, attr_value in self.__dict__.items():
            if callable(attr_value):
                # If it's callable, print the attribute name
                result.append(f"{attr_name}: {attr_value.__name__}")
            else:
                # Otherwise, print the attribute name and its value
                result.append(f"{attr_name}: {attr_value}")
        return "\n".join(result)