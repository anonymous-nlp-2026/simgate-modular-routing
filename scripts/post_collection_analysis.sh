#!/bin/bash
set -e
# conda activate (adjust path for your environment)
cd .

echo "=== Plan-001b Post-Collection Analysis ==="
echo "Date: $(date)"

# 1. Count total episodes per task
echo -e "\n--- Episode counts per task ---"
for dir in results/plan_c/*/; do
  task=$(basename "$dir")
  if [ -f "$dir/episodes.jsonl" ]; then
    count=$(wc -l < "$dir/episodes.jsonl")
    echo "$task: $count episodes"
  else
    echo "$task: MISSING episodes.jsonl"
  fi
done

# 2. Check expected tasks present
echo -e "\n--- Task completeness check ---"
EXPECTED="find-animal identify-life-stages-1 lifespan-longest-lived-then-shortest-lived lifespan-longest-lived measure-melting-point-unknown-substance find-non-living-thing freeze chemistry-mix grow-fruit inclined-plane-determine-angle identify-life-stages-2"
for task in $EXPECTED; do
  if [ -f "results/plan_c/$task/episodes.jsonl" ]; then
    echo "OK $task"
  else
    echo "MISSING $task"
  fi
done

# 3. Check prescreening + mvp reuse data
echo -e "\n--- Reuse data check ---"
for src in results/prescreening results/mvp results/research; do
  count=$(find "$src" -name "episodes.jsonl" 2>/dev/null | wc -l)
  echo "$src: $count task directories"
done

# 4. Total episode count across all sources
echo -e "\n--- Total episodes (all sources) ---"
python3 -c "
import os, json
from collections import defaultdict
counts = defaultdict(int)
for src in ['results/prescreening', 'results/mvp', 'results/research', 'results/plan_c']:
    if not os.path.exists(src): continue
    for task in os.listdir(src):
        f = os.path.join(src, task, 'episodes.jsonl')
        if os.path.isfile(f):
            with open(f) as fh:
                n = sum(1 for _ in fh)
            counts[task] += n
for task in sorted(counts):
    print(f'{task}: {counts[task]} episodes')
print(f'\nTotal tasks: {len(counts)}')
print(f'Total episodes: {sum(counts.values())}')
"

# 5. Data integrity check
echo -e "\n--- Data integrity ---"
python3 -c "
import os, json
errors = []
for src in ['results/plan_c']:
    if not os.path.exists(src): continue
    for task in os.listdir(src):
        f = os.path.join(src, task, 'episodes.jsonl')
        if not os.path.isfile(f): continue
        with open(f) as fh:
            for i, line in enumerate(fh):
                try:
                    d = json.loads(line)
                    assert 'always_internal' in d, 'missing always_internal'
                    assert 'always_deterministic' in d, 'missing always_deterministic'
                    assert 'final_score' in d['always_internal'], 'missing internal final_score'
                    assert 'final_score' in d['always_deterministic'], 'missing det final_score'
                except Exception as e:
                    errors.append(f'{task} line {i}: {e}')
if errors:
    print(f'ERRORS ({len(errors)}):')
    for e in errors[:10]:
        print(f'  {e}')
else:
    print('All episodes valid')
"

echo -e "\n=== Analysis complete ==="
