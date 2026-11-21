# data/ (not in git)

```
data/
├── raw/            read-only original datasets (chmod a-w)
│   └── MOT20/{train,test}/MOT20-XX/{img1,gt,det,seqinfo.ini}
├── weights/        detector weights: YOLO26m fine-tuned on CrowdHuman + CCTV Persons (no MOT17/MOT20)
│   ├── yolo26m_ft1_v3_1280.pt   long Explorer run (May 2026), epoch 160, imgsz 1280  <- used from Step 3
│   ├── yolo26m_ft1_v2_1280.pt   restarted Explorer run, epoch 39, imgsz 1280
│   └── yolo26m_ft1_v1.pt        Colab, 40 epochs, imgsz 1024 (used by Odin's Eye v3)
├── interim/        script outputs that later steps read
│   └── gt_tracks/MOT20/MOT20-XX.parquet   (scripts/00_prepare_data.py, format gt_tracks-1.0)
├── processed/      final tables used for results
└── annotations/    our own labels (later phases)
```

On Explorer the same tree lives under the project directory; only `configs/data/*.yaml`
paths change.
