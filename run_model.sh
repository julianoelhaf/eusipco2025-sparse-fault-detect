#!/bin/bash -l
#SBATCH --job-name=sparse-ml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=./hpc_logs/%x/%x-%j-on-%N.out

mkdir -p ./hpc_logs/$SLURM_JOB_NAME

echo "Your job is running on $(hostname)"

module load python/3.9-anaconda
conda activate sparse_fault_detect # `make create_environment` creates this name

# Adapt these to your own site. On clusters with a small home filesystem and a
# separate capacity filesystem, the MLflow store usually has to live off $HOME,
# which is why the two roots are distinct here; elsewhere they may be the same.
REPO_DIR=/path/to/repos/sparse_fault_detect
MLRUNS_DIR=/path/to/mlflow-store/sparse_fault_detect/mlruns

export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}"

# Find an available port dynamically
MLFLOW_PORT=$(shuf -i 5001-6999 -n 1)
export MLFLOW_TRACKING_URI="http://127.0.0.1:$MLFLOW_PORT"

# Start MLflow server in the background
mlflow server --port $MLFLOW_PORT --backend-store-uri "file://${MLRUNS_DIR}" &
MLFLOW_PID=$! # Store MLflow process ID
sleep 5       # Allow MLflow to initialize

echo "MLflow server started on port $MLFLOW_PORT with PID $MLFLOW_PID"

# Change directory and run the model, passing the MLflow port as an argument.
# Bail out rather than continuing in the submit directory: without the cd the
# run below would resolve a different (or no) copy of the package.
cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} does not exist -- edit it above."; kill $MLFLOW_PID; exit 1; }
python sparse_fault_detect/models/run_model.py --multirun \
    window_extraction.window_length=0.05 \
    training.fault_target=fault_target \
    data_sparsity.zeroing_duration_s=0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045 \
    cluster.port=$MLFLOW_PORT

# After the job finishes, clean up the MLflow server
echo "Shutting down MLflow server..."
kill $MLFLOW_PID
