"""Measure tracking by scored GT occupancy.

HOTA and IDF1 use TrackEval on the observations assigned to each bin, retaining
the sequence timeline and identity equivalence. CLEAR errors and ID switches
are attributed after matching the full sequence, so a switch that crosses a
bin boundary is still counted once, in the destination bin.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from odins_eye.eval.trackeval_runner import COMBINED, mot_challenge_dataset
from odins_eye.io.seqinfo import SeqInfo, read_seqinfo

COUNT_COLUMNS = ["samples", "gt_people", "FN", "FP", "IDSW"]


@dataclass
class _BinnedObservations:
    counts: np.ndarray
    gt_bins: list[np.ndarray]
    tracker_bins: list[np.ndarray]


def _bin_labels(edges: Sequence[int]) -> list[str]:
    if not edges or edges[0] != 0 or any(lo >= hi for lo, hi in pairwise(edges)):
        raise ValueError("Density bin lower bounds must start at zero and strictly increase.")
    labels = [str(lo) if hi == lo + 1 else f"{lo}-{hi - 1}" for lo, hi in pairwise(edges)]
    return [*labels, f"{edges[-1]}+"]


def _cells(boxes: np.ndarray, info: SeqInfo, rows: int, cols: int) -> np.ndarray:
    # TrackEval uses MOT (left, top, width, height). Clip centers to the image grid.
    xy = (boxes[:, :2] + boxes[:, 2:] / 2) / (info.im_width, info.im_height)
    cells = np.floor(xy * (cols, rows)).astype(int).clip(0, (cols - 1, rows - 1))
    return cells[:, 1] * cols + cells[:, 0]


def _clear_matches(
    gt_ids: np.ndarray, tracker_ids: np.ndarray, iou: np.ndarray, previous: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return matched row/column indices using TrackEval CLEAR's continuity rule."""
    if not len(gt_ids) or not len(tracker_ids):
        empty = np.empty(0, dtype=int)
        return empty, empty
    eps = np.finfo(float).eps
    scores = iou + 1000 * (tracker_ids[None, :] == previous[gt_ids, None])
    scores[iou < 0.5 - eps] = 0
    gt_match, tracker_match = linear_sum_assignment(-scores)
    matched = scores[gt_match, tracker_match] > eps
    gt_match, tracker_match = gt_match[matched], tracker_match[matched]
    previous.fill(-1)
    previous[gt_ids[gt_match]] = tracker_ids[tracker_match]
    return gt_match, tracker_match


def _sequence_bins(
    data: dict,
    info: SeqInfo,
    frame_bins: Sequence[int],
    grid_bins: Sequence[int],
    rows: int,
    cols: int,
) -> tuple[_BinnedObservations, _BinnedObservations]:
    frames = _BinnedObservations(np.zeros((len(frame_bins), len(COUNT_COLUMNS)), dtype=int), [], [])
    grids = _BinnedObservations(np.zeros((len(grid_bins), len(COUNT_COLUMNS)), dtype=int), [], [])
    previous = np.full(data["num_gt_ids"], -1)
    last_match = previous.copy()
    for gt_ids, tracker_ids, gt_boxes, tracker_boxes, iou in zip(
        data["gt_ids"],
        data["tracker_ids"],
        data["gt_dets"],
        data["tracker_dets"],
        data["similarity_scores"],
        strict=True,
    ):
        gt_match, tracker_match = _clear_matches(gt_ids, tracker_ids, iou, previous)
        gt_matched_ids, tracker_matched_ids = gt_ids[gt_match], tracker_ids[tracker_match]
        last_ids = last_match[gt_matched_ids]
        switched = gt_match[(last_ids != -1) & (last_ids != tracker_matched_ids)]
        last_match[gt_matched_ids] = tracker_matched_ids

        fn = np.ones(len(gt_ids), dtype=bool)
        fp = np.ones(len(tracker_ids), dtype=bool)
        fn[gt_match] = False
        fp[tracker_match] = False
        frame_bin = np.searchsorted(frame_bins, len(gt_ids), side="right") - 1
        frames.counts[frame_bin] += (1, len(gt_ids), fn.sum(), fp.sum(), len(switched))
        frames.gt_bins.append(np.full(len(gt_ids), frame_bin))
        frames.tracker_bins.append(np.full(len(tracker_ids), frame_bin))

        gt_cells = _cells(gt_boxes, info, rows, cols)
        tracker_cells = _cells(tracker_boxes, info, rows, cols)
        occupancy = np.bincount(gt_cells, minlength=rows * cols)
        cell_bins = np.searchsorted(grid_bins, occupancy, side="right") - 1
        gt_grid_bins = cell_bins[gt_cells]
        tracker_grid_bins = cell_bins[tracker_cells]
        tracker_grid_bins[tracker_match] = gt_grid_bins[gt_match]
        grids.gt_bins.append(gt_grid_bins)
        grids.tracker_bins.append(tracker_grid_bins)
        counts = (
            cell_bins,
            gt_grid_bins,
            gt_grid_bins[fn],
            tracker_grid_bins[fp],
            gt_grid_bins[switched],
        )
        for column, bins in enumerate(counts):
            grids.counts[:, column] += np.bincount(bins, minlength=len(grid_bins))
    return frames, grids


def _compact_ids(frames: list[np.ndarray]) -> tuple[list[np.ndarray], int, int]:
    """Relabel observed IDs densely with a single mapping for the entire bin."""
    flat = np.concatenate(frames)
    unique, ids = np.unique(flat, return_inverse=True)
    offsets = np.cumsum([len(frame) for frame in frames[:-1]])
    return np.split(ids, offsets), len(unique), len(flat)


def _bin_data(data: dict, observations: _BinnedObservations, bin_index: int) -> dict:
    gt_frames, tracker_frames, similarities = [], [], []
    for gt_ids, tracker_ids, iou, gt_bins, tracker_bins in zip(
        data["gt_ids"],
        data["tracker_ids"],
        data["similarity_scores"],
        observations.gt_bins,
        observations.tracker_bins,
        strict=True,
    ):
        gt_keep = gt_bins == bin_index
        tracker_keep = tracker_bins == bin_index
        gt_frames.append(gt_ids[gt_keep])
        tracker_frames.append(tracker_ids[tracker_keep])
        if not (gt_keep.all() and tracker_keep.all()):
            iou = iou[np.ix_(gt_keep, tracker_keep)]
        similarities.append(iou)

    gt_frames, gt_ids, gt_dets = _compact_ids(gt_frames)
    tracker_frames, tracker_ids, tracker_dets = _compact_ids(tracker_frames)
    return {
        "gt_ids": gt_frames,
        "tracker_ids": tracker_frames,
        "similarity_scores": similarities,
        "num_gt_ids": gt_ids,
        "num_tracker_ids": tracker_ids,
        "num_gt_dets": gt_dets,
        "num_tracker_dets": tracker_dets,
        "num_timesteps": data["num_timesteps"],
    }


def evaluate_density_bins(
    trackers_folder: str | Path,
    tracker_name: str,
    gt_folder: Path,
    metrics: pd.DataFrame,
    benchmark: str,
    *,
    frame_bins: Sequence[int],
    grid_bins: Sequence[int],
    rows: int,
    cols: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return frame and grid tables with per-sequence and pooled metrics.

    Use the MOT files written by evaluate_tracks so both passes see the same boxes.
    For the grid, assign a CLEAR-matched prediction to its GT box's bin and an
    unmatched prediction to its own center's bin. TrackEval then scores the
    selected observations with its usual HOTA thresholds and IDF1 matching.
    Every frame and cell, including empty ones, is a sample.
    """
    from trackeval.metrics import HOTA, Identity

    if rows < 1 or cols < 1:
        raise ValueError("Density grid rows and columns must be positive.")
    labels = (_bin_labels(frame_bins), _bin_labels(grid_bins))
    seqs = metrics.index.drop(COMBINED).tolist()
    dataset = mot_challenge_dataset(trackers_folder, tracker_name, gt_folder, seqs, benchmark)
    hota, identity = HOTA(), Identity({"PRINT_CONFIG": False})
    counts_by_scope = ({}, {})
    results_by_scope = ({}, {})
    for seq in seqs:
        raw = dataset.get_raw_seq_data(tracker_name, seq)
        data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
        info = read_seqinfo(gt_folder / seq)
        binned = _sequence_bins(data, info, frame_bins, grid_bins, rows, cols)
        expected = metrics.loc[seq, ["GT_Dets", "FN", "FP", "IDSW"]].to_numpy(dtype=np.int64)
        for scope, (observations, samples) in enumerate(
            zip(binned, (info.seq_length, info.seq_length * rows * cols), strict=True)
        ):
            counts = observations.counts
            if not np.array_equal(counts.sum(axis=0), [samples, *expected]):
                raise RuntimeError(f"{seq}: density totals do not reconcile with TrackEval.")
            counts_by_scope[scope][seq] = counts
            results_by_scope[scope][seq] = []
            for bin_index in range(len(counts)):
                selected = _bin_data(data, observations, bin_index)
                results_by_scope[scope][seq].append(
                    (hota.eval_sequence(selected), identity.eval_sequence(selected))
                )

    out = []
    for counts_by_seq, results, bin_labels in zip(
        counts_by_scope,
        results_by_scope,
        labels,
        strict=True,
    ):
        counts_by_seq[COMBINED] = np.sum(list(counts_by_seq.values()), axis=0)
        results[COMBINED] = [
            (
                hota.combine_sequences({seq: results[seq][i][0] for seq in seqs}),
                identity.combine_sequences({seq: results[seq][i][1] for seq in seqs}),
            )
            for i in range(len(bin_labels))
        ]
        index = pd.MultiIndex.from_product(
            [counts_by_seq, bin_labels], names=["seq", "density_bin"]
        )
        counts = np.concatenate(list(counts_by_seq.values()))
        table = pd.DataFrame(counts, index=index, columns=COUNT_COLUMNS)
        table["HOTA"] = [100 * float(h["HOTA"].mean()) for seq in results.values() for h, _ in seq]
        table["IDF1"] = [100 * float(i["IDF1"]) for seq in results.values() for _, i in seq]
        no_observations = (table["gt_people"] == 0) & (table["FP"] == 0)
        table.loc[no_observations, ["HOTA", "IDF1"]] = np.nan
        out.append(table)
    return out[0], out[1]
