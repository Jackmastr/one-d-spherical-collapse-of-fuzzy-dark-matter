import numpy as np
import matplotlib.pyplot as plt

class SimulationPlotter:
    def __init__(self, results):
        self.results = results

    def plot_global_property(self, property_name, limit_axis=True, ylim=None):
        """Plot the progress of a global property over time."""
        if property_name not in self.results:
            raise ValueError(f"Property '{property_name}' not found in results.")
        
        time = self.results['t']
        property_data = self.results[property_name]
        
        plt.figure(figsize=(10, 6))
        plt.plot(time, property_data)
        plt.xlabel('Time')
        plt.ylabel(property_name.capitalize())
        plt.title(f'{property_name.capitalize()} vs Time')
        plt.grid(True)
        
        self._set_y_limits(property_data, limit_axis, ylim)
        plt.show()

    def plot_shell_property(self, property_name, shell_indices=None, num_shells=5, limit_axis=True, ylim=None, yscale='linear'):
        """Plot the progress of selected shells over time for a given property."""
        if property_name not in self.results:
            raise ValueError(f"Property '{property_name}' not found in results.")
        
        time = self.results['t']
        property_data = self.results[property_name]
        
        if property_data.ndim == 1:
            return self.plot_global_property(property_name, limit_axis, ylim)
        
        if shell_indices is None:
            total_shells = property_data.shape[1]
            shell_indices = np.linspace(0, total_shells-1, min(total_shells, num_shells), dtype=int)
        
        plt.figure(figsize=(10, 6))
        for idx in shell_indices:
            plt.plot(time, property_data[:, idx], label=f'Shell {idx}')
        
        plt.yscale(yscale)
        plt.xlabel('Time')
        plt.ylabel(property_name.capitalize())
        plt.title(f'{property_name.capitalize()} vs Time for Selected Shells')
        plt.legend()
        plt.grid(True)
        
        self._set_y_limits(property_data[:, shell_indices], limit_axis, ylim)
        plt.show()

    def plot_energy_components(self, shell_index=None, limit_axis=True, ylim=None):
        """Plot the progress of different energy components over time."""
        time = self.results['t']
        energy_components = ['e_tot', 'e_g', 'e_k', 'e_r']
        
        plt.figure(figsize=(10, 6))
        all_energy_data = []
        for component in energy_components:
            if component in self.results:
                energy_data = self.results[component]
                if energy_data.ndim > 1 and shell_index is not None:
                    energy_data = energy_data[:, shell_index]
                plt.plot(time, energy_data, label=component.capitalize())
                all_energy_data.extend(energy_data)
        
        plt.xlabel('Time')
        plt.ylabel('Energy')
        title = 'Energy Components vs Time'
        if shell_index is not None:
            title += f' for Shell {shell_index}'
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        self._set_y_limits(all_energy_data, limit_axis, ylim)
        plt.show()

    def plot_normalized_radius(self, shell_index=None, num_shells=None, limit_axis=False, ylim=None):
        """Plot the radius divided by the maximum value r takes for each shell."""
        time = self.results['t']
        r = self.results['r']
        
        max_r = np.max(r, axis=0)
        normalized_r = r / max_r
        
        plt.figure(figsize=(10, 6))
        
        if shell_index is None:
            if num_shells is None:
                shell_indices = range(normalized_r.shape[1])
            else:
                shell_indices = np.linspace(0, normalized_r.shape[1] - 1, num_shells, dtype=int)
        elif isinstance(shell_index, int):
            shell_indices = [shell_index]
        else:
            shell_indices = shell_index
        
        for i in shell_indices:
            plt.plot(time, normalized_r[:, i], label=f'Shell {i}')
        
        plt.xlabel('Time')
        plt.ylabel('Normalized Radius (r / max(r))')
        plt.title('Normalized Radius Over Time')
        plt.legend()
        plt.grid(True)
        
        self._set_y_limits(normalized_r, limit_axis, ylim)
        plt.show()

    def analyze_energy_conservation(self):
        """Analyze and plot energy conservation."""
        t = self.results['t']
        e_tot = np.sum(self.results['e_tot'], axis=1)
        e_k = np.sum(self.results['e_k'], axis=1)
        e_g = np.sum(self.results['e_g'], axis=1)
        e_r = np.sum(self.results['e_r'], axis=1)
        
        e_rel_change = (e_tot - e_tot[0]) / e_tot[0]
        de_dt = np.diff(e_tot) / np.diff(t)
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        axs[0].plot(t, e_tot, label='Total')
        axs[0].plot(t, e_k, label='Kinetic')
        axs[0].plot(t, e_g, label='Gravitational')
        axs[0].plot(t, e_r, label='Rotational')
        axs[0].set_ylabel('Energy')
        axs[0].legend()
        axs[0].set_title('Energy Components')
        
        axs[1].plot(t, e_rel_change)
        axs[1].set_ylabel('Relative Energy Change')
        axs[1].set_title('Relative Total Energy Change')
        
        axs[2].plot(t[1:], de_dt)
        axs[2].set_ylabel('dE/dt')
        axs[2].set_xlabel('Time')
        axs[2].set_title('Energy Change Rate')
        
        plt.tight_layout()
        plt.show()
        
        largest_changes = np.argsort(np.abs(de_dt))[-5:]
        print("Time steps with largest energy changes:")
        for i in largest_changes:
            print(f"Time: {t[i+1]:.6f}, dE/dt: {de_dt[i]:.6e}")

    def _set_y_limits(self, data, limit_axis, ylim):
        if ylim:
            plt.ylim(ylim)
        elif limit_axis:
            lower_percentile = np.percentile(data, 1)
            upper_percentile = np.percentile(data, 99)
            lower_limit = lower_percentile * 0.9 if lower_percentile > 0 else lower_percentile * 1.1
            upper_limit = upper_percentile * 1.1 if upper_percentile > 0 else upper_percentile * 0.9
            plt.ylim(lower_limit, upper_limit)

# Example usage in a Jupyter notebook:
# from plotting import SimulationPlotter
# plotter = SimulationPlotter(results)
# plotter.plot_global_property('e_tot')
# plotter.plot_shell_property('r', num_shells=5)
# plotter.plot_energy_components()
# plotter.plot_normalized_radius(num_shells=5)
# plotter.analyze_energy_conservation()
