#!/bin/bash
# submit_unisdf.sh

if [ -z "$1" ]; then
    echo "Usage: bash submit_unisdf.sh <dataset_name> [gpu_id]"
    echo "Example: bash submit_unisdf.sh Bike 1"
    exit 1
fi

DATASET_NAME=$1
GPU_ID=${2:-0} # Default to GPU 0 if not provided
DATASET_PATH="/scratch/rushikesh/UniSDF-Data/${DATASET_NAME}"
LOG_DIR="/home/rushikesh/NeRF/LOE/UniSDF/${DATASET_NAME}"

# Ensure the log directory exists
mkdir -p "$LOG_DIR"

# Generate a timestamp for unique log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_LOG="${LOG_DIR}/${TIMESTAMP}_output.log"
ERR_LOG="${LOG_DIR}/${TIMESTAMP}_error.log"

echo "Starting background job for dataset: ${DATASET_NAME} on GPU: ${GPU_ID}"
echo "Output logs saving to: ${OUT_LOG}"
echo "Error logs saving to: ${ERR_LOG}"

# Create the background script
cat <<EOT > tmp_job.sh
#!/bin/bash
cd /home/rushikesh/NeRF/UniSDF/unisdf
source venv/bin/activate
# Execute the training pipeline on the selected GPU
bash scripts/run/run_unisdf-colmap_sparse.sh ${DATASET_PATH} ${GPU_ID}
EOT

# Make it executable
chmod +x tmp_job.sh

# Run via nohup in the background to detach from terminal
nohup bash tmp_job.sh > "$OUT_LOG" 2> "$ERR_LOG" &

# Capture the process ID
PID=$!
echo "Job successfully submitted in background! Process ID: $PID"

# Clean up
sleep 1
rm tmp_job.sh
