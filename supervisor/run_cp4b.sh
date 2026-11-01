cd /c/Users/ss864/Desktop/mech-grasp
while pgrep -f recount_all.py >/dev/null 2>&1 || ! grep -q "a2_random {" logs/recount_all.log; do
  grep -q Traceback logs/recount_all.log && break; sleep 30; done
mv runs/T16c runs/T16c_round1; mv runs/T16e runs/T16e_round1; mv runs/T16a runs/T16a_round1
bash harness/chain_cp2.sh T16c T16e T16a > logs/chain_cp4b.log 2>&1
