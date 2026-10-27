cd /c/Users/ss864/Desktop/mech-grasp
until grep -q "CHAIN DONE" logs/chain_cp3.log; do sleep 30; done
bash harness/chain_cp2.sh T13 > logs/chain_cp3b.log 2>&1
