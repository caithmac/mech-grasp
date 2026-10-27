#!/bin/bash
# Checkpoint-2 chain: runs tasks sequentially with full review.
cd /c/Users/ss864/Desktop/mech-grasp
PY=/c/Users/ss864/AppData/Local/miniconda3/envs/mech-grasp/python.exe
for t in "$@"; do
  echo "=== $t start $(date +%H:%M)"
  PYTHONPATH=. $PY -u -m harness.orch run $t --timeout 10800 > logs/$t.log 2>&1
  echo "=== $t exit $? $(date +%H:%M): $(grep -E 'done:|supervisor needed|Traceback' logs/$t.log | tail -1)"
done
echo "=== CHAIN DONE"
