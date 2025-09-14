#!/bin/bash

# Run 5 debates with different seeds in parallel
SEEDS=(7124)

echo "Running 5 parallel debates with different seeds..."

# Run all 5 in parallel, saving output to separate files
for seed in "${SEEDS[@]}"; do
    python cbrn_debate.py --parallel 30 --seed $seed > "seed_${seed}_output.log" 2>&1 &
done

# Wait for all to complete
wait

echo "All debates completed! Check seed_*_output.log files for results."
