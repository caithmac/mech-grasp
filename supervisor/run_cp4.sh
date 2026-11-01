cd /c/Users/ss864/Desktop/mech-grasp
until grep -q "ALL DONE" logs/build_cache4.log 2>/dev/null; do
  grep -q "Traceback" logs/build_cache4.log && { echo "CACHE FAILED" > logs/chain_cp4.log; exit 1; }
  sleep 30
done
bash harness/chain_cp2.sh T16e T16c T16a T16d > logs/chain_cp4.log 2>&1
