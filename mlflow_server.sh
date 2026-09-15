#!/bin/bash -l
#SBATCH --job-name=mlflow-server
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00  # Set a long runtime
#SBATCH --output=./hpc_logs/%x/%x-%j-on-%N.out

mkdir -p ./hpc_logs/$SLURM_JOB_NAME

echo "Starting MLflow server on $(hostname)"

module load python/3.9-anaconda
conda activate juoe_ml # Adjust environment name if needed

export PYTHONPATH="${PYTHONPATH}:/home/hpc/iwi5/<account>/Repositories/sparse_fault_detect"

# Choose a fixed port for consistency
export MLFLOW_PORT=8080
export MLFLOW_TRACKING_URI="http://$(hostname):$MLFLOW_PORT"

# Start MLflow server and keep it alive
mlflow server --host 0.0.0.0 --port $MLFLOW_PORT \
    --backend-store-uri file:///home/woody/iwi5/<account>/Repositories/sparse_fault_detect/mlruns &

wait # Keep job alive as long as MLflow runs
