cd /c/Users/ss864/Desktop/mech-grasp
until [ -f logs/verifies.log ] && grep -q "VERIFIES DONE" logs/verifies.log; do sleep 30; done
mv runs/T15b runs/T15b_round1
bash harness/chain_cp2.sh T15b > logs/chain_cp3e.log 2>&1
