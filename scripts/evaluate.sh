# #!/bin/bash
set -e                                  # stop on error (unchanged)

scripts=$(dirname "$0")
base=$scripts/..

data=$base/sampled_data
configs=$base/configs

translations=$base/translations
mkdir -p "$translations"

# ------------------------------------------------------------------------
src=en
trg=it

num_threads=4
device=0

SECONDS=0                               # measure run-time

model_name=$1
beam_size=${2:-5}             # default 5 (only for bookkeeping)
# ------------------------------------------------------------------------

echo "###############################################################################"
echo "model_name $model_name" beam: "$beam_size" # Note: beam_size from YAML will be used by joeynmt translate

# Choose the correct test source file --------------------------------------
if [[ $model_name == bpe_* ]]; then
  size_suffix_from_model_name=${model_name#bpe_}      # e.g. 2k / 8k / 16k
  size_numeric_for_filename=${size_suffix_from_model_name//k/000} # 2k -> 2000
  test_src=$data/test.bpe${size_numeric_for_filename}.$src
else
  test_src=$data/test.tok.$src
fi
# -------------------------------------------------------------------------

translations_sub=$translations/$model_name
mkdir -p "$translations_sub"

bleu_output_file="$translations_sub/bleu_score.$model_name.txt"

echo "DEBUG: Using test source file: $test_src"
echo "DEBUG: Translations will be saved in: $translations_sub"
echo "DEBUG: Using config file: $configs/$model_name.yaml"
echo "DEBUG: Using reference file: $data/test.tok.$trg for BLEU"
echo "DEBUG: BLEU score will be saved to: $bleu_output_file"

# Basic existence checks ---------------------------------------------------
if [ ! -f "$test_src" ]; then
    echo "ERROR: Test source file not found: $test_src"
    exit 1
fi
if [ ! -f "$configs/$model_name.yaml" ]; then
    echo "ERROR: Config file not found: $configs/$model_name.yaml"
    exit 1
fi
# -------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads \
python -m joeynmt translate "$configs/$model_name.yaml" \
        < "$test_src" \
        > "$translations_sub/test.$model_name.$trg.raw"
# Note: beam size is taken from the YAML testing section.

# Remove BPE “@@” marks ----------------------------------------------------
if [[ -s "$translations_sub/test.$model_name.$trg.raw" ]]; then
    sed -r 's/@@( |$)//g' \
        < "$translations_sub/test.$model_name.$trg.raw" \
        > "$translations_sub/test.$model_name.$trg"
    rm "$translations_sub/test.$model_name.$trg.raw"
else
    echo "WARNING: Raw translation file is empty or not found. Skipping BPE removal and BLEU."
    echo "WARNING: Raw translation file is empty or not found for model $model_name. Cannot calculate BLEU." > "$bleu_output_file"
    exit 1
fi
# -------------------------------------------------------------------------

ref=$data/test.tok.$trg   # tokenised reference

if [ ! -f "$ref" ]; then
    echo "ERROR: Reference file not found: $ref"
    echo "ERROR: Reference file not found: $ref. Cannot calculate BLEU for model $model_name." > "$bleu_output_file"
    exit 1
fi

echo "BLEU:"
sacrebleu "$ref" -m bleu chrf --chrf-word-order 2 -w 4 \
          < "$translations_sub/test.$model_name.$trg" \
          > "$bleu_output_file" 2>&1

echo "BLEU score and details saved to $bleu_output_file"
cat "$bleu_output_file"

echo "time taken: $SECONDS seconds"
