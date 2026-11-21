"""Score MOT files with the ORIGINAL TrackEval (external/TrackEval) and print JSON to stdout.

Run in a subprocess by E0. Upstream still uses `np.float` / `np.int`, which NumPy >= 1.24 removed;
the aliases are restored only inside this process, so the reference code runs unmodified.
"""

import json
import sys
from pathlib import Path

import numpy as np

for _name, _type in {"float": float, "int": int}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _type)

args = json.loads(sys.argv[1])
upstream = Path(args["upstream_dir"]).resolve()
sys.path.insert(0, str(upstream))

import trackeval  # noqa: E402

if not Path(trackeval.__file__).resolve().is_relative_to(upstream):
    raise SystemExit(f"imported the wrong trackeval: {trackeval.__file__}")

from odins_eye.eval.trackeval_runner import evaluate_mot_files  # noqa: E402

df = evaluate_mot_files(
    args["trackers_folder"], args["tracker"], args["gt_folder"], args["seqs"], args["benchmark"]
)
print(df.to_json(orient="split"))
