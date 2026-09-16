#!/bin/bash -l
#SBATCH --job-name=mlflow-server
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00  # Set a long runtime
#SBATCH --output=./hpc_logs/%x/%x-%j-on-%N.out

mkdir -p ./hpc_logs/$SLURM_JOB_NAME

echo "Starting MLflow server on $(hostname)"

module load python/3.9-anaconda
conda activate sparse_fault_detect # `make create_environment` creates this name

# Adapt these to your own site. On clusters with a small home filesystem and a
# separate capacity filesystem, the MLflow store usually has to live off $HOME,
# which is why the two roots are distinct here; elsewhere they may be the same.
REPO_DIR=/path/to/repos/sparse_fault_detect
MLRUNS_DIR=/path/to/mlflow-store/sparse_fault_detect/mlruns

export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}"

# Choose a fixed port for consistency
export MLFLOW_PORT=8080
export MLFLOW_TRACKING_URI="http://$(hostname):$MLFLOW_PORT"

# Start MLflow server and keep it alive
mlflow server --host 0.0.0.0 --port $MLFLOW_PORT \
    --backend-store-uri "file://${MLRUNS_DIR}" &

wait # Keep job alive as long as MLflow runs
