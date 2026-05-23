#!/bin/bash
# Check 72B episode collection progress.
# Shows per-task completion and outputs the --tasks flag for remaining work.

cd "$(dirname "$0")/.." || exit 1

OUTPUT_DIR="results/backbone_72b"
TARGET=${1:-20}

ALL_TASKS=(
    "chemistry-mix"
    "find-animal"
    "find-non-living-thing"
    "freeze"
    "grow-fruit"
    "identify-life-stages-1"
    "identify-life-stages-2"
    "inclined-plane-determine-angle"
    "lifespan-longest-lived"
    "lifespan-longest-lived-then-shortest-lived"
    "measure-melting-point-unknown-substance"
)

echo "72B Episode Collection Progress (target=${TARGET}/task)"
echo "========================================================"
echo ""
printf "  %-50s %5s %6s %10s\n" "Task" "Done" "Target" "Status"
echo "  -------------------------------------------------------------------------"

INCOMPLETE=""
TOTAL_DONE=0
TOTAL_TARGET=0

for task in "${ALL_TASKS[@]}"; do
    episodes_file="${OUTPUT_DIR}/${task}/episodes.jsonl"

    if [ -f "$episodes_file" ]; then
        done=$(wc -l < "$episodes_file" | tr -d ' ')
    else
        done=0
    fi

    remaining=$((TARGET - done))
    if [ $remaining -le 0 ]; then
        status="DONE"
        remaining=0
    else
        status="${remaining} left"
        if [ -z "$INCOMPLETE" ]; then
            INCOMPLETE="${task}"
        else
            INCOMPLETE="${INCOMPLETE},${task}"
        fi
    fi

    printf "  %-50s %5d %6d %10s\n" "$task" "$done" "$TARGET" "$status"
    TOTAL_DONE=$((TOTAL_DONE + done))
    TOTAL_TARGET=$((TOTAL_TARGET + TARGET))
done

echo "  -------------------------------------------------------------------------"
printf "  %-50s %5d %6d\n" "TOTAL" "$TOTAL_DONE" "$TOTAL_TARGET"
echo ""

if [ -z "$INCOMPLETE" ]; then
    echo "All tasks complete!"
else
    echo "Incomplete tasks:"
    echo "  $INCOMPLETE"
    echo ""
    echo "Run on this machine (TP=2):"
    echo "  python src/run_72b_parallel.py --tasks \"$INCOMPLETE\""
    echo ""
    echo "Run on new single-GPU machine (TP=1):"
    echo "  python src/run_72b_parallel.py --tp 1 --tasks \"$INCOMPLETE\""
fi
