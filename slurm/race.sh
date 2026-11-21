# Sourced at the top of a job: the first job of a race to START wins and cancels its siblings.
# RACE_DIR is set by submit_race.sh; jobs submitted directly with sbatch skip this.
if [[ -n "${RACE_DIR:-}" ]]; then
  if mkdir "$RACE_DIR/winner" 2>/dev/null; then           # mkdir is atomic: exactly one winner
    echo "$SLURM_JOB_ID" > "$RACE_DIR/winner/job"
    for _ in $(seq 60); do [[ -s $RACE_DIR/jobs ]] && break; sleep 1; done
    for j in $(cat "$RACE_DIR/jobs" 2>/dev/null); do
      [[ $j != "$SLURM_JOB_ID" ]] && scancel "$j" 2>/dev/null && echo "race won; cancelled sibling $j"
    done
  else
    echo "job $SLURM_JOB_ID: sibling $(cat "$RACE_DIR/winner/job" 2>/dev/null) already started; exiting"
    exit 0
  fi
fi
