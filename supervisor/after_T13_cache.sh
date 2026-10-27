cd /c/Users/ss864/Desktop/mech-grasp
until [ -f logs/chain_cp3b.log ] && grep -q "CHAIN DONE" logs/chain_cp3b.log; do sleep 30; done
PYTHONPATH=. /c/Users/ss864/AppData/Local/miniconda3/envs/mech-grasp/python.exe -u harness/build_cache3.py > logs/build_cache3.log 2>&1
echo "CACHE3 EXIT $?" >> logs/build_cache3.log
