# Explorer HPC jobs

Everything heavy (detection now, training later) runs on Explorer; development stays on the Mac.
The same code runs on every GPU type: `device: auto` picks `cuda:0`, fp16 is used on CUDA.

## One-time setup (login node)

```bash
ssh northeastern
bash <(curl -fsSL https://raw.githubusercontent.com/ozi14/Odin-s-Eye/v4-phase1/slurm/setup_explorer.sh)
```
`setup_explorer.sh` installs uv, clones the repo to `/scratch/$USER/odins-eye`, runs `uv sync`,
and downloads MOT20 there. Upload the detector weights from the Mac:

```bash
cat "data/weights/yolo26m_ft1_v3_1280.pt" | ssh northeastern 'cat > /scratch/$USER/odins-eye/data/weights/yolo26m_ft1_v3_1280.pt'
```

`/scratch` is purged after inactivity: everything there must be reproducible (code from git, data
from the official site), and results are copied back to the Mac right after a run.

## Running a job: first free GPU wins (H200 or A100)

```bash
cd /scratch/$USER/odins-eye && git pull --ff-only && uv sync
slurm/submit_race.sh slurm/detect.sbatch                                  # full Step 3
slurm/submit_race.sh slurm/detect.sbatch max_frames=32 data.sequences=[MOT20-01]   # smoke test
squeue -u $USER
```
`submit_race.sh` submits one copy per GPU type (`--gres=gpu:h200:1`, `--gres=gpu:a100:1`), each to
`gpu-short,gpu`; the first copy to start cancels the others. Never request a bare `gpu:1`: it can
land on a V100/P100, which our torch build (CUDA 13) does not support.

Logs: `slurm/logs/<job-name>-<jobid>.out`. Copy results back to the Mac:

```bash
rsync -av northeastern:/scratch/ucar.t/odins-eye/data/interim/detections/ data/interim/detections/
rsync -av northeastern:/scratch/ucar.t/odins-eye/experiments/ experiments/
```
