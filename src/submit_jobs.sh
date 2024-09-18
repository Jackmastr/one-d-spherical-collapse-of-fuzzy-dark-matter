#!/bin/bash

# Function to submit a job
submit_job() {
    local config_file=$1
    local config_number=$(basename "$config_file" .json | sed 's/config_//')
    local job_name="sim_${config_number}"

    cmd="python run_simulation.py --config $config_file"

    tmp_job_script=$(mktemp)
    cat << EOF > "$tmp_job_script"
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

$cmd
EOF

    sbatch "$tmp_job_script"
    rm "$tmp_job_script"

    echo "Submitted job: $cmd"
}

# Ensure the logs directory exists
mkdir -p logs

# Generate config files
python generate_configs.py

# Submit jobs for all config files
for config_file in configs/*.json; do
    submit_job "$config_file"
done