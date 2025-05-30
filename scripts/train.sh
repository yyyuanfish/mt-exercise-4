#! /bin/bash
set -e                                  # exit on error

scripts=$(dirname "$0")
base=$scripts/..

configs="$base/configs"
# models="$base/models"
logs="$base/logs"

mkdir -p "$logs"



# 1 argument = model name (YAML file without .yaml)
model_name=$1
[[ -z $model_name ]] && { echo "Usage: bash scripts/train.sh <model_name>"; exit 1; }

# num_threads=8
# detect logical CPU cores (macOS & Linux both available) -------------------------
if command -v nproc &>/dev/null; then
  num_threads=$(nproc)                          # Linux for my teammate
else
  num_threads=$(sysctl -n hw.logicalcpu)        # macOS
fi
echo "Using $num_threads CPU threads"
# ------------------------------------------------------------------

# measure time

SECONDS=0

#logs=$base/logs

# log dir 
log_dir="$logs/$model_name"
mkdir -p "$log_dir"

#model_name=?
#mkdir -p $logs
#mkdir -p $logs/$model_name

#OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/$model_name.yaml > $logs/$model_name/out 2> $logs/$model_name/err
OMP_NUM_THREADS=$num_threads python -m joeynmt train "$configs/$model_name.yaml" > "$log_dir/out" 2> "$log_dir/err"

echo "time taken: $SECONDS seconds"
