import numpy as np
from numba import jit
from abc import ABC, abstractmethod

def const_rho_func(self):
    return self.m_tot / (4/3 * np.pi * self.r_max**3)

def const_j_func(self):
    return self.j_coef

def gmr_j_func(self):
    return self.j_coef * np.sqrt(self.G * self.m_enc * self.r_ta)

def r_is_r_ta_func(self):
        return self.r

def hubble_v_func(self):
    return self.H * self.r

def keep_edges_shell_vol_func(self):
    # Calculate the volumes of spherical shells
    r_inner = np.zeros_like(self.r)
    r_inner[1:] = self.r[:-1]
    volumes = 4/3 * np.pi * (self.r**3 - r_inner**3)
    return volumes

class SimulationComponent(ABC):
    @abstractmethod
    def __call__(self, sim):
        pass

class StepperStrategy(SimulationComponent):
    pass

class VelocityVerletStepper(StepperStrategy):
    def __call__(self, sim):
        print(f"VelocityVerletStepper.__call__ called with sim: {sim}")
        sim.r = self._velocity_verlet_numba(sim.r, sim.v, sim.a, sim.dt)
        sim.handle_reflections()
        sim.m_enc = sim.m_enc_func()
        a_old = sim.a.copy()
        sim.a = sim.a_func()
        sim.v = self._velocity_verlet_update_v_numba(sim.v, a_old, sim.a, sim.dt)
        sim.t += sim.dt

    @staticmethod
    @jit(nopython=True)
    def _velocity_verlet_numba(r, v, a, dt):
        return r + v * dt + 0.5 * a * dt**2

    @staticmethod
    @jit(nopython=True)
    def _velocity_verlet_update_v_numba(v, a_old, a_new, dt):
        return v + 0.5 * (a_old + a_new) * dt

class BeemanStepper(StepperStrategy):
    def __call__(self, sim):
        if sim.a_prev is None:
            # Use Taylor expansion for the first step
            sim.r = sim.r + sim.v * sim.dt + 0.5 * sim.a * sim.dt**2
            sim.handle_reflections()
            sim.m_enc = sim.m_enc_func()
            a_new = sim.a_func()
            v_new = sim.v + sim.a * sim.dt
        else:
            sim.r = self._beeman_r_numba(sim.r, sim.v, sim.a, sim.a_prev, sim.dt)
            sim.handle_reflections()
            sim.m_enc = sim.m_enc_func()
            a_new = sim.a_func()
            v_new = self._beeman_v_numba(sim.v, sim.a, a_new, sim.a_prev, sim.dt)
        
        # Update for next step
        sim.a_prev = sim.a
        sim.a = a_new
        sim.v = v_new
        sim.t += sim.dt

    @staticmethod
    @jit(nopython=True)
    def _beeman_r_numba(r, v, a, a_prev, dt):
        return r + v * dt + (4 * a - a_prev) * (dt**2) / 6

    @staticmethod
    @jit(nopython=True)
    def _beeman_v_numba(v, a, a_new, a_prev, dt):
        return v + (2 * a_new + 5 * a - a_prev) * dt / 6

class StepperFactory:
    @staticmethod
    def create(stepper_name):
        if stepper_name == "velocity_verlet":
            return VelocityVerletStepper()
        elif stepper_name == "beeman":
            return BeemanStepper()
        else:
            raise ValueError(f"Unknown stepper: {stepper_name}")


class TurnaroundFunctions:
    @staticmethod
    def r_is_r_ta_func(self):
        return self.r
    
    @staticmethod
    def r_ta_analytic(self):
        return self.r_ta


# class Steppers:
#     @staticmethod
#     def velocity_verlet(self):
#         self.r = Steppers._velocity_verlet_numba(self.r, self.v, self.a, self.dt)
#         self.handle_reflections()
#         self.m_enc = self.m_enc_func()
#         a_old = self.a.copy()
#         self.a = self.a_func()
#         self.v = Steppers._velocity_verlet_update_v_numba(self.v, a_old, self.a, self.dt)
#         self.t += self.dt
    
#     velocity_verlet.__doc__ = "Velocity Verlet: r = r + vdt + (1/2)adt^2, v = (1/2)(a_i + a_{i-1})dt"

#     @staticmethod
#     @jit(nopython=True)
#     def _velocity_verlet_numba(r, v, a, dt):
#         return r + v * dt + 0.5 * a * dt**2

#     @staticmethod
#     @jit(nopython=True)
#     def _velocity_verlet_update_v_numba(v, a_old, a_new, dt):
#         return v + 0.5 * (a_old + a_new) * dt
    
#     @staticmethod
#     def beeman(self):
#         # Beeman's algorithm implementation
#         if self.a_prev is None:
#             # Use Taylor expansion for the first step
#             self.r = self.r + self.v * self.dt + 0.5 * self.a * self.dt**2
#             self.handle_reflections()
#             self.m_enc = self.m_enc_func()
#             a_new = self.a_func()
#             v_new = self.v + self.a * self.dt
#         else:
#             self.r = Steppers._beeman_r_numba(self.r, self.v, self.a, self.a_prev, self.dt)
#             self.handle_reflections()
#             self.m_enc = self.m_enc_func()
#             a_new = self.a_func()
#             v_new = Steppers._beeman_v_numba(self.v, self.a, a_new, self.a_prev, self.dt)
        
#         # Update for next step
#         self.a_prev = self.a
#         self.a = a_new
#         self.v = v_new
#         self.t += self.dt
    
#     beeman.__doc__ = "Beeman: r = r + v * dt + (4 * a - a_prev) * (dt**2) / 6, v = v + (2 * a_new + 5 * a - a_prev) * dt / 6"

#     @staticmethod
#     @jit(nopython=True)
#     def _beeman_r_numba(r, v, a, a_prev, dt):
#         return r + v * dt + (4 * a - a_prev) * (dt**2) / 6

#     @staticmethod
#     @jit(nopython=True)
#     def _beeman_v_numba(v, a, a_new, a_prev, dt):
#         return v + (2 * a_new + 5 * a - a_prev) * dt / 6

class AccelerationFunctions:
    @staticmethod
    def soft_grav_a_func(self):
        r_soft = self.soft_func()
        return AccelerationFunctions._soft_grav_a_func_numba(self.G, self.m_enc, self.j, self.r, r_soft)
    
    soft_grav_a_func.__doc__ = "a = -G * m_enc / r_soft**2 + j**2 / r**3"

    @staticmethod
    @jit(nopython=True)
    def _soft_grav_a_func_numba(G, m_enc, j, r, r_soft):
        return -G * m_enc / r_soft**2 + j**2 / r**3
    
    @staticmethod
    def soft_all_a_func(self):
        r_soft = self.soft_func()
        return AccelerationFunctions._soft_all_a_func_numba(self.G, self.m_enc, self.j, r_soft)
    
    soft_all_a_func.__doc__ = "-G * m_enc / r_soft**2 + j**2 / r_soft**3"

    @staticmethod
    @jit(nopython=True)
    def _soft_all_a_func_numba(G, m_enc, j, r_soft):
        return -G * m_enc / r_soft**2 + j**2 / r_soft**3


class SofteningFunctions:
    @staticmethod
    def const_soft_func(self):
        return SofteningFunctions._const_soft_func_numba(self.r, self.softlen)
    
    const_soft_func.__doc__ = "r = sqrt(r^2 + softlen^2)"

    @staticmethod
    @jit(nopython=True)
    def _const_soft_func_numba(r, softlen):
        return np.sqrt(r**2 + softlen**2)

    @staticmethod
    def r_ta_soft_func(self):
        return SofteningFunctions._r_ta_soft_func_numba(self.r, self.softlen, self.r_ta)
    
    r_ta_soft_func.__doc__ = "r = sqrt(r^2 + (softlen * r_ta)^2)"

    @staticmethod
    @jit(nopython=True)
    def _r_ta_soft_func_numba(r, softlen, r_ta):
        return np.sqrt(r**2 + (softlen * r_ta)**2)
    
class EnclosedMassFunctions:
    @staticmethod
    def m_enc_inclusive(self):
        self.thickness_func()
        return EnclosedMassFunctions._m_enc_inclusive_numba(self.r, self.m)
    
    m_enc_inclusive.__doc__ = "M_enc,r = sum_{r' <= r} m'"

    @staticmethod
    @jit(nopython=True)
    def _m_enc_inclusive_numba(r, m):
        # Sort radii and get sorting indices
        sorted_indices = np.argsort(r)
        # Sort masses based on radii
        sorted_masses = m[sorted_indices]
        # Calculate cumulative sum of sorted masses
        cumulative_mass = np.cumsum(sorted_masses)
        # Create the result array and assign values using advanced indexing
        m_enc = np.empty_like(cumulative_mass)
        m_enc[sorted_indices] = cumulative_mass
        return m_enc
    
    @staticmethod
    def m_enc_overlap_inclusive(self):
        self.thickness_func()
        return EnclosedMassFunctions._m_enc_overlap_inclusive_numba(self.r, self.m, self.thicknesses)
    
    m_enc_overlap_inclusive.__doc__ = "Inclusive enclosed mass function with shell overlap"

    @staticmethod
    @jit(nopython=True)
    def _m_enc_overlap_inclusive_numba(r, m, thicknesses):
        n = len(r)
        m_enc = np.zeros_like(m)
        
        # Pre-compute inner and outer radii
        inner_radii = r - thicknesses
        outer_radii = r
        
        # Pre-compute volumes
        volumes = outer_radii**3 - inner_radii**3
        
        # Compute enclosed mass for each shell
        for i in range(n):
            # Include own mass
            m_enc[i] = m[i]
            for j in range(n):
                if i == j:
                    continue
                
                if r[i] > r[j]:
                    # No overlap, include full mass
                    m_enc[i] += m[j]
                elif r[j] - thicknesses[j] < r[i]:
                    # Partial overlap, calculate enclosed mass fraction
                    overlap_volume = min(r[i]**3 - (r[j] - thicknesses[j])**3, volumes[j])
                    volume_fraction = overlap_volume / volumes[j]
                    m_enc[i] += m[j] * volume_fraction
        
        return m_enc

class EnergyFunctions:
    @staticmethod
    def default_energy_func(self):
        self.e_k, self.e_g, self.e_r, self.e_tot = EnergyFunctions._default_energy_func_numba(self.G, self.m, self.v, self.m_enc, self.r, self.j)
    
    default_energy_func.__doc__ = "Default energy calculation function"

    @staticmethod
    @jit(nopython=True)
    def _default_energy_func_numba(G, m, v, m_enc, r, j):
        e_k = 0.5 * m * v**2
        e_g = -G * m * m_enc / r
        e_r = 0.5 * m * j**2 / r**2
        e_tot = e_k + e_g + e_r
        return e_k, e_g, e_r, e_tot
    
class TimeScaleFunctions:
    @staticmethod
    def dynamical_and_zero_timescale_func(self):
        self.t_dyn, self.t_zero = TimeScaleFunctions._dynamical_and_zero_timescale_numba(self.G, self.m_enc, self.r, self.v)
        self.min_time_scale = min(self.t_dyn, self.t_zero)
    
    dynamical_and_zero_timescale_func.__doc__ = "Dynamical time and time to reach r=0 calculation"
    
    @staticmethod
    @jit(nopython=True)
    def _dynamical_and_zero_timescale_numba(G, m_enc, r, v):
        t_dyn = np.min(1/np.sqrt(G * m_enc / r**3))
        
        # Calculate time to reach r=0 for each shell
        t_zero = np.inf
        for i in range(len(r)):
            if v[i] < 0:  # Only consider inward-moving shells
                t = r[i] / abs(v[i])
                if t < t_zero:
                    t_zero = t
        
        return t_dyn, t_zero
    
    @staticmethod
    def dynamical_and_rmin_timescale_func(self):
        self.t_dyn, self.t_rmin = TimeScaleFunctions._dynamical_and_rmin_timescale_numba(self.G, self.m_enc, self.r, self.v, self.r_min)
        self.min_time_scale = min(self.t_dyn, self.t_rmin)
    
    dynamical_and_rmin_timescale_func.__doc__ = "Dynamical time and time to reach r_min calculation"
    
    @staticmethod
    @jit(nopython=True)
    def _dynamical_and_rmin_timescale_numba(G, m_enc, r, v, r_min):
        t_dyn = np.min(1/np.sqrt(G * m_enc / r**3))
        
        # Calculate time to reach r_min for each shell
        t_rmin = np.inf
        for i in range(len(r)):
            if v[i] < 0:  # Only consider inward-moving shells
                t = (r[i] - r_min) / abs(v[i])
                if t < t_rmin:
                    t_rmin = t
        
        return t_dyn, t_rmin

    
    @staticmethod
    def rubin_loeb_timescale_func(self):
        self.t_dyn, self.t_vel, self.t_acc = TimeScaleFunctions._rubin_loeb_timescale_numba(self.G, self.m_enc, self.r, self.v, self.a, self.r_max)
        self.min_time_scale = min(self.t_dyn, self.t_vel, self.t_acc)
    
    rubin_loeb_timescale_func.__doc__ = "Rubin-Loeb timescale calculation"
    
    @staticmethod
    @jit(nopython=True)
    def _rubin_loeb_timescale_numba(G, m_enc, r, v, a, r_max):
        eps = 1e-2
        t_dyn = np.min(1/np.sqrt(G * m_enc / r**3))
        t_vel = np.min(r_max / (np.abs(v)+eps))
        t_acc = np.min(np.sqrt(r_max / (np.abs(a)+eps)))
        return t_dyn, t_vel, t_acc
    
    @staticmethod
    def rubin_loeb_cross_timescale_func(self):
        self.t_dyn, self.t_vel, self.t_acc, self.t_cross = TimeScaleFunctions._rubin_loeb_cross_timescale_numba(self.G, self.m_enc, self.r, self.v, self.a, self.r_max)
        self.min_time_scale = min(self.t_dyn, self.t_vel, self.t_acc, self.t_cross)

    rubin_loeb_cross_timescale_func.__doc__ = "Rubin-Loeb timescale calculation with crossing time"

    @staticmethod
    @jit(nopython=True)
    def _rubin_loeb_cross_timescale_numba(G, m_enc, r, v, a, r_max):
        eps = 1e-2
        t_dyn = np.min(1/np.sqrt(G * m_enc / r**3))
        t_vel = np.min(r_max / (np.abs(v)+eps))
        t_acc = np.min(np.sqrt(r_max / (np.abs(a)+eps)))
        
        # Calculate t_cross: time for any two shells to reach the same position
        n = len(r)
        t_cross = np.inf
        for i in range(n):
            for j in range(i+1, n):
                dr = np.abs(r[i] - r[j])
                dv = np.abs(v[i] - v[j])
                if dv > 1e-9:
                    t = dr / dv
                    if t > 0 and t < t_cross:
                        t_cross = t
        
        return t_dyn, t_vel, t_acc, t_cross
    
class TimeStepFunctions:
    @staticmethod
    def const_timestep(self):
        pass
    
    const_timestep.__doc__ = "Constant timestep function"

    @staticmethod
    def simple_adaptive_timestep(self):
        self.dt = max(self.dt_min, TimeStepFunctions._simple_adaptive_timestep_numba(self.safety_factor, self.min_time_scale))
    
    simple_adaptive_timestep.__doc__ = "Simple adaptive timestep function"

    @staticmethod
    @jit(nopython=True)
    def _simple_adaptive_timestep_numba(safety_factor, min_time_scale):
        return safety_factor * min_time_scale

    
class InitialDensityProfileFunctions:
    @staticmethod
    def const_rho_func(self):
        return InitialDensityProfileFunctions._const_rho_func_numba(self.r_max, self.m_tot)
    
    const_rho_func.__doc__ = "Constant density profile function"

    @staticmethod
    @jit(nopython=True)
    def _const_rho_func_numba(r_max, m_tot):
        return m_tot / (4/3 * np.pi * r_max**3)

    @staticmethod
    def power_law_rho_func(self):
        return InitialDensityProfileFunctions._power_law_rho_func_numba(self.r, self.r_max, self.m_tot, self.gamma)
    
    power_law_rho_func.__doc__ = "Power-law density profile function"

    @staticmethod
    @jit(nopython=True)
    def _power_law_rho_func_numba(r, r_max, m_tot, gamma):
        norm_const = (3 + gamma) * m_tot / (4 * np.pi * r_max**(3 + gamma))
        return norm_const * r**gamma
    
    @staticmethod
    def background_plus_power_law_rho_func(self):
        return InitialDensityProfileFunctions._background_plus_power_law_rho_func_numba(self.r, self.rho_bar, self.m_tot, self.gamma, self.r_max)
    
    background_plus_power_law_rho_func.__doc__ = "Background plus power-law density profile function"

    @staticmethod
    @jit(nopython=True)
    def _background_plus_power_law_rho_func_numba(r, rho_bar, m_tot, gamma, r_max):
        return rho_bar + (3 + gamma) * m_tot / (4 * np.pi * r_max**(3 + gamma)) * r**gamma
    
class ShellThicknessFunction:
    @staticmethod
    def const_shell_thickness(self):
        self.thicknesses = ShellThicknessFunction._const_shell_thickness_numba(self.r, self.thickness_coef)
    
    const_shell_thickness.__doc__ = "Constant shell thickness function"

    @staticmethod
    @jit(nopython=True)
    def _const_shell_thickness_numba(r, thickness_coef):
        return np.full(len(r), thickness_coef)