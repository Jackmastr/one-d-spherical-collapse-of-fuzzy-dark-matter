import argparse
import json
from collapse import SphericalCollapse
from utils import save_to_hdf5, load_simulation_data

def run_simulation(config, output_file):
    sim = SphericalCollapse(config)
    sim.run()
    save_to_hdf5(sim, output_file)
    print(f"Simulation completed. Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run SphericalCollapse simulation")
    parser.add_argument("--config", type=str, required=True, help="JSON file containing simulation parameters")
    parser.add_argument("--output", type=str, help="Output HDF5 file name (overrides config file)")

    args = parser.parse_args()

    # Load config from file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Override output file if specified in command line
    if args.output:
        config['output_file'] = args.output

    run_simulation(config, config['output_file'])

    # Optionally, load and verify the saved data
    params, snapshots = load_simulation_data(config['output_file'])
    print(f"Loaded {len(snapshots['t'])} snapshots from {config['output_file']}")

if __name__ == "__main__":
    main()