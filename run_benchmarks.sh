#!/bin/bash
# Run all AoC 2025 benchmarks sequentially (one at a time to avoid contention)

MODELS="haiku sonnet opus"
DAYS="1 2 3 4 5 6 7 8 9 10 11 12"

# Optional: specify which models/days to run via args
if [ "$1" != "" ]; then
    MODELS="$1"
fi
if [ "$2" != "" ]; then
    DAYS="$2"
fi

echo "Running benchmarks for models: $MODELS"
echo "Running benchmarks for days: $DAYS"
echo "=========================================="

for model in $MODELS; do
    echo ""
    echo "=== Model: $model ==="
    for day in $DAYS; do
        echo ""
        echo "--- Day $day with $model ---"
        python -m aoc2025 benchmark "$day" --model "$model"
        echo ""
    done
done

echo ""
echo "=========================================="
echo "All benchmarks complete!"
