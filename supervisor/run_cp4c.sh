export PYTHONPATH=.
for s in "sup_T16e.py" "sup_T16a.py" "recount_all.py a3 a3_random" "verify_T16d.py"; do
  echo "=== $s start $(date +%H:%M)"; python supervisor/$s > logs/$(echo $s|cut -d. -f1).log 2>&1; echo "=== $s exit $? $(date +%H:%M)"
done
echo "=== CP4C DONE"
