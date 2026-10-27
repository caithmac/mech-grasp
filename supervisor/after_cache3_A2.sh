cd /c/Users/ss864/Desktop/mech-grasp
until [ -f logs/build_cache3.log ] && grep -q "CACHE3 EXIT" logs/build_cache3.log; do sleep 30; done
bash harness/chain_cp2.sh T15a T15b T15d T15c > logs/chain_cp3c.log 2>&1
