import json
import os
from itertools import product

def generate_config(N=2, r_min=1e-2, softlen=0, safety_factor=1e-3, thickness_coef=0, j_coef=0, outfile=None):
    if outfile is None:
        outfile = f"runs/sim_N{N}_soft{softlen}_sf{safety_factor}_tc{thickness_coef}_jc{j_coef}.h5"
    else:
        outfile = outfile
    config = {
        "N": N,
        "softlen": softlen,
        "safety_factor": safety_factor,
        "thickness_coef": thickness_coef,
        "j_coef": j_coef,
        "t_max": 5,
        "r_min": r_min,
        "m_tot": 1,
        "stepper_strategy": "beeman",
        "output_file": outfile
    }
    return config

def main():
    # Ensure the configs directory exists
    os.makedirs("configs", exist_ok=True)

    # Parameter combinations
    softlen_values = [0, 1e-4, 1e-3, 1e-2, 1e-1]
    safety_factor_values = [1e-2, 1e-4, 1e-5]
    j_coef_values = [1e-3, 1e-2, 1e-1, 5e-1]
    thickness_coef_values = [0, 1e-3, 1e-2, 1e-1]
    r_min_values = [1e-3, 1e-1]

    configs = [generate_config()]

    for s in softlen_values:
        configs.append(generate_config(softlen=s, outfile=f"runs/softlen/{s}.h5"))

    for sf in safety_factor_values:
        configs.append(generate_config(safety_factor=sf, outfile=f"runs/safety_factor/{sf}.h5"))

    for jc in j_coef_values:
        configs.append(generate_config(j_coef=jc, outfile=f"runs/j_coef/{jc}.h5"))

    for rm in r_min_values:
        configs.append(generate_config(r_min=rm, outfile=f"runs/r_min/{rm}.h5"))

    for tc in thickness_coef_values:
        configs.append(generate_config(thickness_coef=tc, outfile=f"runs/thickness_coef/{tc}.h5"))

    # # Generate all combinations
    # for N, softlen, safety_factor, thickness_coef, j_coef in product(
    #     N_values, softlen_values, safety_factor_values, thickness_coef_values, j_coef_values
    # ):
    #     configs.append(generate_config(
    #         N=N,
    #         softlen=softlen,
    #         safety_factor=safety_factor,
    #         thickness_coef=thickness_coef,
    #         j_coef=j_coef
    #     ))

    # Save configurations
    for i, config in enumerate(configs):
        filename = f"config_{i+1}.json"
        with open(os.path.join("configs", filename), "w") as f:
            json.dump(config, f, indent=2)

    print(f"Generated {len(configs)} configuration files.")


if __name__ == "__main__":
    main()