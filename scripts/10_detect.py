"""Step 3: run the detector on every frame of every sequence -> detections-1.0 Parquet.

Outputs
  data/interim/detections/<detector>/<seq>.parquet      (…_smoke/ when max_frames is set)
  experiments/<date>_det_<detector>/                    config, git, env (GPU), timing, summary
"""

from __future__ import annotations

import hashlib
import json

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from odins_eye.io.schemas import DETECTIONS_SCHEMA, write_table
from odins_eye.perception.detector import DetectorConfig, YoloDetector, detect_sequence
from odins_eye.utils.env import runtime_info
from odins_eye.utils.paths import PROJECT_ROOT
from odins_eye.utils.run import new_run_dir


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@hydra.main(version_base=None, config_path="../configs", config_name="detect")
def main(cfg: DictConfig) -> None:
    smoke = cfg.max_frames is not None
    desc = cfg.experiment.desc + ("_smoke" if smoke else "")
    run = new_run_dir(cfg.experiment.id, desc, cfg)

    d = OmegaConf.to_container(cfg.detector, resolve=True)
    weights = PROJECT_ROOT / d["weights"]
    digest = sha256(weights)
    if digest != d["weights_sha256"]:
        raise SystemExit(
            f"{weights.name}: sha256 {digest[:12]}… != config {d['weights_sha256'][:12]}…"
        )
    params = {k: v for k, v in d.items() if k != "weights_sha256"}
    params |= {"weights": str(weights), "classes": tuple(params["classes"])}
    detector = YoloDetector(DetectorConfig(**params))

    env = runtime_info() | {"detector": detector.settings(), "weights_sha256": digest}
    (run / "env.json").write_text(json.dumps(env, indent=2, default=str) + "\n")
    print(
        f"device={detector.device} half={detector.half} gpu={env['gpu']} end2end={detector.end2end}"
    )

    out_dir = PROJECT_ROOT / cfg.out_dir / desc
    raw = PROJECT_ROOT / cfg.data.raw_dir / cfg.data.split
    rows = []
    for seq in cfg.data.sequences:
        dets, stats = detect_sequence(detector, raw / seq, cfg.max_frames)
        write_table(dets, out_dir / f"{seq}.parquet", DETECTIONS_SCHEMA)
        rows.append(stats)
        print(json.dumps(stats))
    summary = pd.DataFrame(rows)
    summary.to_csv(run / "timing.csv", index=False)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "out_dir": str(out_dir.relative_to(PROJECT_ROOT)),
                "gpu": env["gpu"],
                "sequences": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))
    rel = out_dir.relative_to(PROJECT_ROOT)
    print(f"\ndetections -> {rel}/   run -> {run.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
