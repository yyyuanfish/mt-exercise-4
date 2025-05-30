#!/usr/bin/env bash
###############################################################################
# PREPARE EN→IT DATA FOR JOEYNMT
# Steps:
#   1. 100k random sub-sample from train set
#   2. copy dev / test as-is
#   3. Sacremoses tokenisation
#   4. replace HTML &apos; with apostrophe
#   5. learn joint BPE (2000 / 8000 / 16000)
#   6. build joint vocab
#   7. apply BPE to train / dev / test
#
# Re-running is safe: existing artefacts are kept.
###############################################################################
set -e

#############################  1) PARAMETERS  #################################
PAIR=en-it            # file-name stub in raw data
SRC=en                # source language   (English)
TRG=it                # target language   (Italian)

# Auto-detect raw folder (supports both data/ and scripts/data/)
if [[ -f data/train.${PAIR}.${SRC} ]]; then
    RAW=data
elif [[ -f scripts/data/train.${PAIR}.${SRC} ]]; then
    RAW=scripts/data
else
    echo "ERROR: cannot find IWSLT files train.${PAIR}.${SRC}" >&2
    exit 1
fi
echo "RAW directory  : $RAW"

WORK=sampled_data     # all outputs go here
NUM=100000            # training pairs to keep
BPE_SIZES=(2000 8000 16000)
mkdir -p "$WORK"

#############################  2) SHUF COMMAND  ###############################
# GNU `shuf` exists on Linux; macOS gets it via coreutils as `gshuf`.
if command -v shuf  &>/dev/null; then
    SHUF_CMD=shuf
elif command -v gshuf &>/dev/null; then
    SHUF_CMD=gshuf
else
    echo "ERROR: neither shuf nor gshuf available" >&2
    exit 1
fi

##################  3) SAMPLING & TOKENISATION BLOCK  #########################
# Skip this whole block if tokenised train already exists
if [[ ! -f "$WORK/train.tok.$SRC" ]]; then
  echo "=== 1) Sub-sampling $NUM EN–IT sentence pairs ==="
  paste "$RAW/train.$PAIR.$SRC" "$RAW/train.$PAIR.$TRG" | \
    ${SHUF_CMD} -n "$NUM" > "$WORK/train.tsv"

  cut -f1 "$WORK/train.tsv" > "$WORK/train.$SRC"   # English
  cut -f2 "$WORK/train.tsv" > "$WORK/train.$TRG"   # Italian
  rm "$WORK/train.tsv"

  echo "=== 2) Copying dev / test splits ==="
  for SPLIT in dev test; do
    cp "$RAW/$SPLIT.$PAIR.$SRC" "$WORK/$SPLIT.$SRC"
    cp "$RAW/$SPLIT.$PAIR.$TRG" "$WORK/$SPLIT.$TRG"
  done

  echo "=== 3) Sacremoses tokenisation ==="
  for SPLIT in train dev test; do
    sacremoses -l "$SRC" tokenize < "$WORK/$SPLIT.$SRC" > "$WORK/$SPLIT.tok.$SRC"
    sacremoses -l "$TRG" tokenize < "$WORK/$SPLIT.$TRG" > "$WORK/$SPLIT.tok.$TRG"
  done
  echo "Tokenisation finished."

  echo "=== 4) Replacing &apos; → ' in *.tok files ==="
  for SPLIT in train dev test; do
    for LANG in $SRC $TRG; do
      sed -i.bak "s/&apos;/'/g" "$WORK/$SPLIT.tok.$LANG" && rm "$WORK/$SPLIT.tok.$LANG.bak"
    done
  done
fi

######################  4) LEARN + APPLY BPE PER SIZE  ########################
echo "=== 5) Learning & applying joint BPE ==="
for SIZE in "${BPE_SIZES[@]}"; do
  CODES="$WORK/bpe${SIZE}.codes"
  VOCAB_SRC="$WORK/vocab${SIZE}.${SRC}"
  VOCAB_TRG="$WORK/vocab${SIZE}.${TRG}"
  VOCAB_JOINT="$WORK/vocab${SIZE}.joint"

  # learn codes if missing
  if [[ ! -f "$CODES" ]]; then
    echo "Learning BPE-$SIZE"
    subword-nmt learn-joint-bpe-and-vocab --total-symbols \
      -s "$SIZE" \
      --input "$WORK/train.tok.$SRC" "$WORK/train.tok.$TRG" \
      -o "$CODES" \
      --write-vocabulary "$VOCAB_SRC" "$VOCAB_TRG"

    cat "$VOCAB_SRC" "$VOCAB_TRG" | cut -f1 -d' ' | sort -u > "$VOCAB_JOINT"
  fi

  # apply to each split
  echo "Applying BPE-$SIZE"
  for SPLIT in train dev test; do
    for LANG in $SRC $TRG; do
      OUTFILE="$WORK/$SPLIT.bpe${SIZE}.$LANG"
      [[ -f "$OUTFILE" ]] && { echo "  $OUTFILE exists – skip"; continue; }

      subword-nmt apply-bpe -c "$CODES" \
        --vocabulary "$WORK/vocab${SIZE}.${LANG}" \
        --vocabulary-threshold 0 \
        < "$WORK/$SPLIT.tok.$LANG" \
        > "$OUTFILE"
    done
  done
  echo "BPE-$SIZE complete"
done

echo "All preprocessing finished."
echo "Tokenised files : sampled_data/* .tok.*"
echo "BPE outputs     : sampled_data/* .bpe{2000,8000,16000}.*"

