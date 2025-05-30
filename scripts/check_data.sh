#!/usr/bin/env bash
echo "=== basic sanity checks ==="
for SPLIT in train dev test; do
  echo "# $SPLIT"
  wc -l sampled_data/$SPLIT.en sampled_data/$SPLIT.it
done
echo -e "\nEmpty-line check:"
grep -n "^$" sampled_data/train.en | head || true

echo -e "\nToken sample:"
paste sampled_data/train.en sampled_data/train.it | head -n 2 | column -t -s$'\t'

echo -e "\n<unk> in BPE2000:"
grep -o "<unk>" sampled_data/train.bpe2000.en | wc -l

echo -e "\nDev chrF quick smoke test:"
sacrebleu -m chrf -l en-it sampled_data/dev.it < sampled_data/dev.en


