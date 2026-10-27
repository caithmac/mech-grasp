cd /c/Users/ss864/Desktop/mech-grasp
PY=/c/Users/ss864/AppData/Local/miniconda3/envs/mech-grasp/python.exe
until [ -f logs/chain_cp3c.log ] && grep -q "CHAIN DONE" logs/chain_cp3c.log; do sleep 30; done
for v in verify_T14a verify_T14b verify_T13; do
  echo "=== $v $(date +%H:%M)"; PYTHONPATH=. $PY -u supervisor/$v.py > logs/$v.log 2>&1; echo "=== $v exit $?"
done
echo VERIFIES DONE
