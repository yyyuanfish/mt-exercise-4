#!/bin/bash
# File: beam_experiment.sh - Modified for Joey NMT that doesn't support beam-size CLI args

# Add environment check
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" == "base" ]; then
    echo "ERROR: Please activate conda environment first!"
    echo "Run: conda activate joeynmt-env"
    exit 1
fi

PROJECT_ROOT="/Users/liuxduan/Desktop/mt-exercise-4-master"
BASE_CONFIG="$PROJECT_ROOT/configs/bpe_8k.yaml"
MODEL_FILE="models/bpe_8k/best.ckpt"
TEST_SRC="sampled_data/test.bpe8000.en"
OUTPUT_DIR="$PROJECT_ROOT/beam_results"
REF_FILE="$PROJECT_ROOT/sampled_data/test.tok.it"
CONFIG_DIR="$PROJECT_ROOT/configs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CONFIG_DIR/temp"

# Check if necessary files exist
echo "Checking files..."
if [ ! -f "$BASE_CONFIG" ]; then
    echo "ERROR: Base config file not found: $BASE_CONFIG"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/$TEST_SRC" ]; then
    echo "ERROR: Test source file not found: $PROJECT_ROOT/$TEST_SRC"
    exit 1
fi

if [ ! -f "$REF_FILE" ]; then
    echo "ERROR: Reference file not found: $REF_FILE"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/$MODEL_FILE" ]; then
    echo "ERROR: Model file not found: $PROJECT_ROOT/$MODEL_FILE"
    exit 1
fi

echo "All files found. Starting beam search experiment..."

# Clear old results
> "$OUTPUT_DIR/all_results.txt"

# Test different beam sizes
for beam in 1 2 4 8 16 32 64 128 256 512; do
    echo "----------------------------------------"
    echo "Running beam size $beam..."
    START_TIME=$(date +%s)
    echo "Started at: $(date -r $START_TIME)"

    # Create a temporary config file for each beam size
    TEMP_CONFIG="$CONFIG_DIR/temp/bpe_8k_beam${beam}.yaml"
    cp "$BASE_CONFIG" "$TEMP_CONFIG"

    # Change config file for beam size
    if grep -q "beam_size:" "$TEMP_CONFIG"; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/beam_size: [0-9]*/beam_size: $beam/" "$TEMP_CONFIG"
        else
            sed -i "s/beam_size: [0-9]*/beam_size: $beam/" "$TEMP_CONFIG"
        fi
        echo "Modified existing beam_size to $beam"
    else
        echo "" >> "$TEMP_CONFIG"
        echo "# Added for beam search experiment" >> "$TEMP_CONFIG"
        echo "testing:" >> "$TEMP_CONFIG"
        echo "  beam_size: $beam" >> "$TEMP_CONFIG"
        echo "Added beam_size: $beam to config"
    fi

    cd "$PROJECT_ROOT"

    SUCCESS=false
    echo "Running translation with beam size $beam..."
    python -m joeynmt translate "$TEMP_CONFIG" \
        --ckpt "$MODEL_FILE" \
        --output-path "beam_results/pred_beam${beam}.it" \
        < "$TEST_SRC" 2>&1 | tee "beam_results/beam${beam}.log"

    if [ -f "beam_results/pred_beam${beam}.it" ]; then
        SUCCESS=true
        echo "Translation completed successfully"
    else
        echo "Retrying without explicit checkpoint..."
        python -m joeynmt translate "$TEMP_CONFIG" \
            --output-path "beam_results/pred_beam${beam}.it" \
            < "$TEST_SRC" 2>&1 | tee "beam_results/beam${beam}_retry.log"

        if [ -f "beam_results/pred_beam${beam}.it" ]; then
            SUCCESS=true
            echo "Translation completed successfully (without explicit checkpoint)"
        fi
    fi

    END_TIME=$(date +%s)
    echo "Ended at:   $(date -r $END_TIME)"
    ELAPSED=$((END_TIME - START_TIME))
    HUMAN_TIME=$(printf '%02dh:%02dm:%02ds' $((ELAPSED/3600)) $(( (ELAPSED%3600)/60 )) $((ELAPSED%60)))

    if [ "$SUCCESS" = true ] && [ -s "beam_results/pred_beam${beam}.it" ]; then
        sacrebleu "$REF_FILE" -i "beam_results/pred_beam${beam}.it" -m bleu -b -w 4 > "beam_results/bleu_beam${beam}.txt"
        BLEU_SCORE=$(cat "beam_results/bleu_beam${beam}.txt")
        echo "$beam $HUMAN_TIME $BLEU_SCORE" >> "$OUTPUT_DIR/all_results.txt"
        echo "Beam size $beam completed - Time: $HUMAN_TIME, BLEU: $BLEU_SCORE"
    else
        echo "$beam FAILED translation_error_or_empty_output" >> "$OUTPUT_DIR/all_results.txt"
        echo "ERROR: Translation failed or output is empty for beam size $beam"
    fi

    rm -f "$TEMP_CONFIG"
done

echo "========================================"
echo "Experiment completed!"
echo "Results summary:"
if [ -f "$OUTPUT_DIR/all_results.txt" ]; then
    echo -e "BeamSize\tTime\t\tBLEU"
    awk '{printf "%-8s\t%-10s\t%s\n", $1, $2, $3}' "$OUTPUT_DIR/all_results.txt"
    echo ""
    echo "Detailed results saved in: $OUTPUT_DIR/"
else
    echo "No results file found. Check logs in $OUTPUT_DIR/"
fi