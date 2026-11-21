# data/ (not in git)

```
data/
├── raw/            read-only original datasets (chmod a-w)
│   └── MOT20/{train,test}/MOT20-XX/{img1,gt,det,seqinfo.ini}
├── weights/        model weights, e.g. yolo26m_crowdhuman_cctv_v1.pt (v3 fine-tune, 2026-03-30)
├── interim/        script outputs that later steps read
│   └── gt_tracks/MOT20/MOT20-XX.parquet   (scripts/00_prepare_data.py, format gt_tracks-1.0)
├── processed/      final tables used for results
└── annotations/    our own labels (later phases)
```

On Explorer the same tree lives under the project directory; only `configs/data/*.yaml`
paths change.
