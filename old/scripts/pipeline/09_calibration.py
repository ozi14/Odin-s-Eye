"""
09_calibration.py — Phase 0: Offline Calibration Setup

Parses WILDTRACK camera calibration XMLs and computes:
  1. Projection matrices P = K × [R|t] for each camera
  2. Ground-plane homographies H = K × [r₁, r₂, t] (Z=0 simplification)
  3. FOV polygons on the ground plane (clipped to WILDTRACK bounds)
  4. Pairwise overlap ratios between all camera pairs
  5. Overlap graph + maximal cliques for cross-camera association

Run once. Outputs are cached to output/calibration_cache.json and
a BEV visualization is saved to output/fov_bev.png.

Usage:
    python scripts/pipeline/09_calibration.py
    python scripts/pipeline/09_calibration.py --threshold 0.10
    python scripts/pipeline/09_calibration.py --ablation   # runs 0.01 → 0.30

Reference:
    WILDTRACK dataset — Chavdarova et al., CVPR 2018
    Calibration format follows the official WILDTRACK-toolkit:
    https://github.com/Chavdarova/WILDTRACK-toolkit
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import networkx as nx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# WILDTRACK ground plane definition (from official toolkit)
GROUND_ORIGIN_CM = (-300.0, -90.0)   # (x_min, y_min) in centimeters
GROUND_SIZE_CM   = (3600.0, 1200.0)  # (width, height) = 36m × 12m
GRID_STEP_CM     = 2.5               # 2.5 cm per grid cell

# Derived bounds for clipping
GROUND_X_MIN = GROUND_ORIGIN_CM[0]                          # -300
GROUND_X_MAX = GROUND_ORIGIN_CM[0] + GROUND_SIZE_CM[0]      #  3300
GROUND_Y_MIN = GROUND_ORIGIN_CM[1]                          # -90
GROUND_Y_MAX = GROUND_ORIGIN_CM[1] + GROUND_SIZE_CM[1]      #  1110

# WILDTRACK image resolution
IMG_W, IMG_H = 1920, 1080

# Camera file mapping: our canonical ID → file prefix
CAMERA_MAP = {
    "C1": "CVLab1",
    "C2": "CVLab2",
    "C3": "CVLab3",
    "C4": "CVLab4",
    "C5": "IDIAP1",
    "C6": "IDIAP2",
    "C7": "IDIAP3",
}

# WILDTRACK ground-plane bounding box as a Shapely polygon for clipping
GROUND_BOUNDS = box(GROUND_X_MIN, GROUND_Y_MIN, GROUND_X_MAX, GROUND_Y_MAX)


# ---------------------------------------------------------------------------
# XML Parsing
# ---------------------------------------------------------------------------
def load_intrinsic(xml_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse an intrinsic calibration XML (OpenCV format).
    
    Returns:
        K:    (3, 3) camera matrix
        dist: (5, 1) distortion coefficients
    """
    tree = ET.parse(xml_path)
    
    # Camera matrix
    cm = tree.find("camera_matrix")
    rows = int(cm.find("rows").text)
    cols = int(cm.find("cols").text)
    data = cm.find("data").text.strip().split()
    K = np.array([float(x) for x in data], dtype=np.float64).reshape(rows, cols)
    
    # Distortion coefficients
    dc = tree.find("distortion_coefficients")
    rows_d = int(dc.find("rows").text)
    cols_d = int(dc.find("cols").text)
    data_d = dc.find("data").text.strip().split()
    dist = np.array([float(x) for x in data_d], dtype=np.float64).reshape(rows_d, cols_d)
    
    return K, dist


def load_extrinsic(xml_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse an extrinsic calibration XML.
    
    The XML contains:
      - rvec: Rodrigues rotation vector (3 values)
      - tvec: Translation vector (3 values, in centimeters)
    
    Returns:
        rvec: (3, 1) rotation vector
        tvec: (3, 1) translation vector
    """
    tree = ET.parse(xml_path)
    
    rvec_text = tree.find("rvec").text.strip().split()
    tvec_text = tree.find("tvec").text.strip().split()
    
    rvec = np.array([float(x) for x in rvec_text], dtype=np.float64).reshape(3, 1)
    tvec = np.array([float(x) for x in tvec_text], dtype=np.float64).reshape(3, 1)
    
    return rvec, tvec


def load_all_calibrations(calib_dir: str) -> dict:
    """
    Load intrinsic (zero-distortion) and extrinsic calibrations for all 7 cameras.
    
    Args:
        calib_dir: Path to the wildtrack_calibrations directory
        
    Returns:
        Dictionary mapping camera ID → {K, dist, rvec, tvec}
    """
    calib_dir = Path(calib_dir)
    intr_dir = calib_dir / "intrinsic_zero"
    extr_dir = calib_dir / "extrinsic"
    
    cameras = {}
    for cam_id, prefix in CAMERA_MAP.items():
        intr_path = intr_dir / f"intr_{prefix}.xml"
        extr_path = extr_dir / f"extr_{prefix}.xml"
        
        if not intr_path.exists():
            raise FileNotFoundError(f"Intrinsic file not found: {intr_path}")
        if not extr_path.exists():
            raise FileNotFoundError(f"Extrinsic file not found: {extr_path}")
        
        K, dist = load_intrinsic(str(intr_path))
        rvec, tvec = load_extrinsic(str(extr_path))
        
        cameras[cam_id] = {
            "K": K,
            "dist": dist,
            "rvec": rvec,
            "tvec": tvec,
        }
        
    print(f"✅ Loaded calibrations for {len(cameras)} cameras")
    return cameras


# ---------------------------------------------------------------------------
# Projection Math
# ---------------------------------------------------------------------------
def compute_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """
    Convert Rodrigues rotation vector to 3×3 rotation matrix.
    
    The Rodrigues vector encodes rotation axis (direction) and angle (magnitude):
        axis  = rvec / |rvec|
        angle = |rvec|  (in radians)
    
    cv2.Rodrigues() uses the Rodrigues formula to expand this into R.
    """
    R, _ = cv2.Rodrigues(rvec)
    return R


def compute_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build the 3×4 projection matrix: P = K × [R | t]
    
    This maps homogeneous 3D world points [X, Y, Z, 1]^T
    to homogeneous 2D pixel coordinates [su, sv, s]^T.
    
    Args:
        K: (3, 3) intrinsic camera matrix
        R: (3, 3) rotation matrix  
        t: (3, 1) translation vector
        
    Returns:
        P: (3, 4) projection matrix
    """
    Rt = np.hstack([R, t])  # (3, 4) — [R | t]
    P = K @ Rt               # (3, 4) — K × [R | t]
    return P


def compute_ground_homography(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute the 3×3 homography for mapping between the ground plane (Z=0)
    and the image plane.
    
    When Z = 0, the 3rd column of R is irrelevant:
        H = K × [r₁, r₂, t]
    
    This gives us:
        H:    world (X, Y) → pixel (u, v)     [forward projection]
        H⁻¹:  pixel (u, v) → world (X, Y)     [back-projection]
    
    Args:
        K: (3, 3) intrinsic camera matrix
        R: (3, 3) rotation matrix
        t: (3, 1) translation vector
        
    Returns:
        H: (3, 3) ground-plane homography
    """
    # Extract r1 (column 0), r2 (column 1) from R, and flatten t
    r1 = R[:, 0:1]  # (3, 1)
    r2 = R[:, 1:2]  # (3, 1)
    
    # H = K × [r1 | r2 | t]
    H = K @ np.hstack([r1, r2, t])  # (3, 3)
    
    return H


def pixel_to_world(H_inv: np.ndarray, u: float, v: float) -> tuple[float, float]:
    """
    Back-project a pixel (u, v) to the ground plane (X, Y) in centimeters.
    
    Uses the inverse homography H⁻¹ to map from image space to ground space.
    
    The math:
        [sX]       [u]
        [sY] = H⁻¹ [v]
        [s ]       [1]
        
        X = sX / s,  Y = sY / s
    """
    pixel_h = np.array([u, v, 1.0], dtype=np.float64)
    world_h = H_inv @ pixel_h
    
    # Dehomogenize: divide by the 3rd coordinate
    if abs(world_h[2]) < 1e-10:
        return float('inf'), float('inf')
    
    X = world_h[0] / world_h[2]
    Y = world_h[1] / world_h[2]
    
    return float(X), float(Y)


def world_to_pixel(H: np.ndarray, X: float, Y: float) -> tuple[float, float]:
    """
    Project a ground-plane point (X, Y) in cm to pixel coordinates (u, v).
    
    Uses the forward homography H to map from ground space to image space.
    """
    world_h = np.array([X, Y, 1.0], dtype=np.float64)
    pixel_h = H @ world_h
    
    if abs(pixel_h[2]) < 1e-10:
        return float('inf'), float('inf')
    
    u = pixel_h[0] / pixel_h[2]
    v = pixel_h[1] / pixel_h[2]
    
    return float(u), float(v)


# ---------------------------------------------------------------------------
# FOV Polygon Computation
# ---------------------------------------------------------------------------
def compute_fov_polygon(H: np.ndarray, img_w: int = IMG_W, img_h: int = IMG_H,
                         grid_step: float = 25.0) -> Polygon | None:
    """
    Compute a camera's field-of-view polygon on the ground plane.
    
    APPROACH: Instead of back-projecting image boundary pixels to the ground
    (which fails for steep-angle cameras where top-of-image rays project to
    infinity), we use the FORWARD direction:
    
      1. Generate a dense grid of ground plane points within WILDTRACK bounds
      2. Forward-project each ground point INTO the camera using H
      3. Keep only ground points whose projection lands inside the image
      4. Compute the convex hull of valid ground points → FOV polygon
    
    This is the same strategy used by the official WILDTRACK toolkit.
    
    Args:
        H: (3, 3) forward homography (world → pixel)
        img_w: image width in pixels
        img_h: image height in pixels
        grid_step: spacing between test ground points in cm (smaller = more precise)
        
    Returns:
        Shapely Polygon representing the camera's FOV on the ground, or None
    """
    from shapely.geometry import MultiPoint
    
    visible_ground_points = []
    
    # Generate ground grid points within WILDTRACK bounds
    xs = np.arange(GROUND_X_MIN, GROUND_X_MAX + grid_step, grid_step)
    ys = np.arange(GROUND_Y_MIN, GROUND_Y_MAX + grid_step, grid_step)
    
    for X in xs:
        for Y in ys:
            # Forward-project: ground (X, Y) → pixel (u, v)
            u, v = world_to_pixel(H, X, Y)
            
            # Check if the projected point lands inside the image
            if 0 <= u < img_w and 0 <= v < img_h:
                visible_ground_points.append((X, Y))
    
    if len(visible_ground_points) < 3:
        return None
    
    # Build convex hull of all visible ground points
    try:
        multi_pt = MultiPoint(visible_ground_points)
        hull = multi_pt.convex_hull
        
        if hull.is_empty or hull.area < 1.0:
            return None
        
        # Clip to WILDTRACK ground bounds (should already be within, but safety)
        clipped = hull.intersection(GROUND_BOUNDS)
        
        if clipped.is_empty or clipped.area < 1.0:
            return None
        
        return clipped
    except Exception as e:
        print(f"  ⚠️ Polygon construction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Overlap Graph & Cliques
# ---------------------------------------------------------------------------
def compute_pairwise_overlaps(fov_polygons: dict) -> dict:
    """
    Compute overlap ratios for all camera pairs.
    
    Overlap ratio = intersection_area / min(area_i, area_j)
    
    Using min() instead of union() makes the metric asymmetry-aware:
    if a small camera's FOV is entirely inside a large camera's FOV,
    the overlap is 100%, which is correct for tracking purposes.
    
    Returns:
        Dictionary: {(ci, cj): overlap_ratio} for all pairs where both have valid FOVs
    """
    cam_ids = sorted(fov_polygons.keys())
    overlaps = {}
    
    for i, ci in enumerate(cam_ids):
        for j, cj in enumerate(cam_ids):
            if j <= i:
                continue
            
            poly_i = fov_polygons[ci]
            poly_j = fov_polygons[cj]
            
            if poly_i is None or poly_j is None:
                overlaps[(ci, cj)] = 0.0
                continue
            
            intersection = poly_i.intersection(poly_j)
            min_area = min(poly_i.area, poly_j.area)
            
            if min_area < 1e-6:
                overlaps[(ci, cj)] = 0.0
            else:
                overlaps[(ci, cj)] = intersection.area / min_area
    
    return overlaps


def build_overlap_graph(overlaps: dict, threshold: float = 0.05) -> tuple[nx.Graph, list]:
    """
    Build an undirected overlap graph and find maximal cliques.
    
    Nodes = cameras, Edges = camera pairs with overlap > threshold.
    Maximal cliques represent groups of cameras that ALL see the same area.
    
    These cliques define "communication groups" — within a clique, cross-camera
    identity association is meaningful because the same person CAN appear
    in all cameras of the clique.
    
    Uses Bron-Kerbosch algorithm via NetworkX.
    
    Args:
        overlaps: pairwise overlap ratios from compute_pairwise_overlaps()
        threshold: minimum overlap ratio to add an edge
        
    Returns:
        G: NetworkX graph
        cliques: list of lists, each inner list is a maximal clique
    """
    G = nx.Graph()
    G.add_nodes_from(sorted(set(
        cam for pair in overlaps.keys() for cam in pair
    )))
    
    for (ci, cj), ratio in overlaps.items():
        if ratio >= threshold:
            G.add_edge(ci, cj, weight=ratio)
    
    cliques = [sorted(c) for c in nx.find_cliques(G)]
    cliques.sort(key=lambda c: (-len(c), c[0]))  # Largest first
    
    return G, cliques


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Premium dark color palette for cameras
CAM_COLORS = {
    "C1": "#FF4D6A",  # Coral red
    "C2": "#00D2FF",  # Electric cyan
    "C3": "#7B68EE",  # Medium slate blue
    "C4": "#FFB347",  # Pastel orange
    "C5": "#00E5A0",  # Emerald green
    "C6": "#FF69B4",  # Hot pink
    "C7": "#87CEEB",  # Sky blue
}

DARK_BG    = "#1a1a2e"
DARK_PANEL = "#16213e"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR  = "#e0e0e0"


def _draw_polygon(ax, poly, color, alpha=0.25, linewidth=1.5, linestyle='-', zorder=2):
    """Helper: draw a Shapely polygon on a matplotlib axis."""
    if poly is None or poly.is_empty:
        return
    if poly.geom_type == 'MultiPolygon':
        geoms = list(poly.geoms)
    else:
        geoms = [poly]
    for geom in geoms:
        x, y = geom.exterior.xy
        ax.fill(x, y, alpha=alpha, color=color, zorder=zorder)
        ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, 
                zorder=zorder + 1)


def visualize_fov_bev(fov_polygons: dict, overlaps: dict, cliques: list,
                      threshold: float, output_path: str):
    """
    Premium dark-themed bird's-eye view of camera FOVs with overlap regions.
    
    - Individual camera FOVs drawn with distinct colors
    - Overlap density shown via heatmap (how many cameras see each region)
    - Pairwise intersection regions highlighted
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from shapely.ops import unary_union
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 9),
                              gridspec_kw={'width_ratios': [3, 2]})
    fig.patch.set_facecolor(DARK_BG)
    
    # ── Left panel: Individual FOVs ──
    ax1 = axes[0]
    ax1.set_facecolor(DARK_PANEL)
    
    # Draw ground plane
    ground_rect = plt.Rectangle(
        (GROUND_X_MIN, GROUND_Y_MIN), 
        GROUND_SIZE_CM[0], GROUND_SIZE_CM[1],
        linewidth=1.5, edgecolor='#4a4a6a', facecolor=DARK_PANEL, 
        linestyle='--', zorder=0
    )
    ax1.add_patch(ground_rect)
    
    # Draw each camera's FOV
    for cam_id, poly in sorted(fov_polygons.items()):
        if poly is None:
            continue
        color = CAM_COLORS[cam_id]
        _draw_polygon(ax1, poly, color, alpha=0.15, linewidth=2.0, zorder=2)
        
        # Label at centroid
        cx, cy = poly.centroid.x, poly.centroid.y
        area_m2 = poly.area / 10000
        ax1.text(cx, cy, f"{cam_id}\n{area_m2:.0f}m²", fontsize=9, fontweight='bold',
                ha='center', va='center', color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color, 
                          edgecolor='white', alpha=0.85, linewidth=0.5), zorder=10)
    
    # Draw scale bar
    ax1.plot([GROUND_X_MIN + 50, GROUND_X_MIN + 550], 
             [GROUND_Y_MAX + 60, GROUND_Y_MAX + 60], 
             color=TEXT_COLOR, linewidth=3, zorder=10)
    ax1.text(GROUND_X_MIN + 300, GROUND_Y_MAX + 100, "5 m", 
             fontsize=9, color=TEXT_COLOR, ha='center', zorder=10)
    
    ax1.set_title("Camera Fields of View", fontsize=14, fontweight='bold', 
                  color=TEXT_COLOR, pad=10)
    ax1.set_xlabel("X (cm)", fontsize=10, color=TEXT_COLOR)
    ax1.set_ylabel("Y (cm)", fontsize=10, color=TEXT_COLOR)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.15, color=GRID_COLOR)
    ax1.set_xlim(GROUND_X_MIN - 150, GROUND_X_MAX + 150)
    ax1.set_ylim(GROUND_Y_MIN - 150, GROUND_Y_MAX + 200)
    ax1.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color('#4a4a6a')
    
    # ── Right panel: Overlap density heatmap ──
    ax2 = axes[1]
    ax2.set_facecolor(DARK_PANEL)
    
    # Compute coverage count: how many cameras see each ground point
    grid_res = 50  # cm resolution for heatmap
    xs = np.arange(GROUND_X_MIN, GROUND_X_MAX, grid_res)
    ys = np.arange(GROUND_Y_MIN, GROUND_Y_MAX, grid_res)
    from shapely.geometry import Point
    
    coverage = np.zeros((len(ys), len(xs)))
    valid_polys = {cid: p for cid, p in fov_polygons.items() if p is not None}
    
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            pt = Point(x + grid_res/2, y + grid_res/2)
            count = sum(1 for p in valid_polys.values() if p.contains(pt))
            coverage[yi, xi] = count
    
    # Custom colormap: dark → blue → cyan → yellow → white
    cmap_colors = ['#1a1a2e', '#0d3b66', '#1e90ff', '#00e5a0', '#ffd700', '#ffffff']
    cmap = LinearSegmentedColormap.from_list('coverage', cmap_colors, N=256)
    
    im = ax2.imshow(coverage, extent=[GROUND_X_MIN, GROUND_X_MAX, GROUND_Y_MIN, GROUND_Y_MAX],
                    origin='lower', cmap=cmap, vmin=0, vmax=7, aspect='equal',
                    interpolation='bilinear')
    
    # Overlay FOV outlines
    for cam_id, poly in sorted(fov_polygons.items()):
        if poly is None:
            continue
        color = CAM_COLORS[cam_id]
        _draw_polygon(ax2, poly, color, alpha=0, linewidth=1.2, linestyle='--', zorder=5)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
    cbar.set_label("Camera Coverage Count", fontsize=10, color=TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    cbar.ax.yaxis.label.set_color(TEXT_COLOR)
    
    ax2.set_title("Overlap Density (cameras per point)", fontsize=14, 
                  fontweight='bold', color=TEXT_COLOR, pad=10)
    ax2.set_xlabel("X (cm)", fontsize=10, color=TEXT_COLOR)
    ax2.set_ylabel("Y (cm)", fontsize=10, color=TEXT_COLOR)
    ax2.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('#4a4a6a')
    
    # Suptitle
    n_edges = sum(1 for v in overlaps.values() if v >= threshold)
    fig.suptitle(
        f"WILDTRACK — Bird's Eye View  |  {n_edges}/21 overlapping pairs  |  "
        f"{len(cliques)} clique(s)  |  threshold = {threshold:.0%}",
        fontsize=15, fontweight='bold', color='white', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"✅ BEV visualization saved to: {output_path}")


def visualize_overlap_matrix(overlaps: dict, output_path: str):
    """Dark-themed pairwise overlap heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    cam_ids = sorted(CAMERA_MAP.keys())
    n = len(cam_ids)
    matrix = np.zeros((n, n))
    
    for (ci, cj), ratio in overlaps.items():
        i = cam_ids.index(ci)
        j = cam_ids.index(cj)
        matrix[i, j] = ratio
        matrix[j, i] = ratio
    
    np.fill_diagonal(matrix, 1.0)
    
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_PANEL)
    
    # Custom colormap
    cmap = LinearSegmentedColormap.from_list('overlap', 
        ['#1a1a2e', '#0d3b66', '#1e90ff', '#00e5a0', '#ffd700', '#ff4d6a'], N=256)
    
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cam_ids, fontsize=12, fontweight='bold', color=TEXT_COLOR)
    ax.set_yticklabels(cam_ids, fontsize=12, fontweight='bold', color=TEXT_COLOR)
    
    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = '#1a1a2e' if val > 0.7 else 'white'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center', 
                    fontsize=11, color=color, fontweight='bold')
    
    ax.set_title("Pairwise FOV Overlap", fontsize=16, fontweight='bold',
                 color='white', pad=15)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Overlap Ratio", fontsize=11, color=TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)
    
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color('#4a4a6a')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"✅ Overlap matrix saved to: {output_path}")


def visualize_per_camera_overlaps(cameras_processed: dict, fov_polygons: dict,
                                   output_dir: str):
    """
    For each camera, generate an image showing which overlap regions exist.
    
    This projects the pairwise overlap polygons from the ground plane back 
    into each camera's pixel space, creating color-coded overlay masks.
    
    These regions tell the VLM: "this part of your frame is also seen by 
    cameras X, Y, Z — look for identity matches there."
    
    Output:
        output/{cam_id}_overlap_regions.png — one per camera
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    
    valid_polys = {cid: p for cid, p in fov_polygons.items() if p is not None}
    
    for cam_id, cam in sorted(cameras_processed.items()):
        if fov_polygons.get(cam_id) is None:
            continue
        
        H_inv = cam["H_inv"]  # world → but we need world → pixel, so use H
        H = cam["H"]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor('#0a0a1a')
        
        # Draw image boundary
        img_rect = plt.Rectangle((0, 0), IMG_W, IMG_H, linewidth=2,
                                  edgecolor='#4a4a6a', facecolor='#0a0a1a', zorder=0)
        ax.add_patch(img_rect)
        
        my_poly = fov_polygons[cam_id]
        overlap_count = 0
        
        for other_id, other_poly in sorted(valid_polys.items()):
            if other_id == cam_id:
                continue
            
            # Compute ground-plane intersection
            intersection = my_poly.intersection(other_poly)
            if intersection.is_empty or intersection.area < 100:  # < 1m²
                continue
            
            overlap_count += 1
            color = CAM_COLORS[other_id]
            
            # Get intersection polygon coordinates on ground plane
            if intersection.geom_type == 'MultiPolygon':
                geoms = list(intersection.geoms)
            else:
                geoms = [intersection]
            
            for geom in geoms:
                ground_coords = list(geom.exterior.coords)
                
                # Project each ground coordinate into this camera's pixel space
                pixel_coords = []
                for gx, gy in ground_coords:
                    u, v = world_to_pixel(H, gx, gy)
                    # Clamp to image bounds
                    u = max(0, min(IMG_W, u))
                    v = max(0, min(IMG_H, v))
                    pixel_coords.append((u, v))
                
                if len(pixel_coords) >= 3:
                    # Draw the overlap region
                    poly_patch = MplPolygon(pixel_coords, closed=True,
                                           facecolor=color, edgecolor=color,
                                           alpha=0.2, linewidth=1.5, zorder=3)
                    ax.add_patch(poly_patch)
                    
                    # Draw border
                    poly_border = MplPolygon(pixel_coords, closed=True,
                                            fill=False, edgecolor=color,
                                            linewidth=1.0, linestyle='--', 
                                            alpha=0.6, zorder=4)
                    ax.add_patch(poly_border)
                    
                    # Label this overlap region
                    cx = np.mean([p[0] for p in pixel_coords])
                    cy = np.mean([p[1] for p in pixel_coords])
                    # Only label if centroid is inside image
                    if 50 < cx < IMG_W - 50 and 50 < cy < IMG_H - 50:
                        ratio_key = (min(cam_id, other_id), max(cam_id, other_id))
                        ax.text(cx, cy, f"↔ {other_id}", fontsize=8,
                                fontweight='bold', color='white', ha='center',
                                va='center', alpha=0.9,
                                bbox=dict(boxstyle='round,pad=0.2', 
                                          facecolor=color, alpha=0.7,
                                          edgecolor='none'), zorder=6)
        
        # Camera label
        ax.text(IMG_W / 2, 40, f"{cam_id} — Overlap Regions ({overlap_count} cameras)",
                fontsize=14, fontweight='bold', color='white', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=CAM_COLORS[cam_id],
                          alpha=0.85, edgecolor='white', linewidth=0.5), zorder=10)
        
        ax.set_xlim(0, IMG_W)
        ax.set_ylim(IMG_H, 0)  # Flip Y (image convention: top-left origin)
        ax.set_aspect('equal')
        ax.axis('off')
        
        out_path = os.path.join(output_dir, f"{cam_id}_overlap_regions.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        plt.close()
    
    print(f"✅ Per-camera overlap regions saved to: {output_dir}/C*_overlap_regions.png")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_round_trip(cameras_processed: dict, n_test_points: int = 20):
    """
    Validation: pixel → world → pixel round-trip error.
    
    For random pixels within the image, back-project to the ground plane,
    then forward-project back. The error should be < 1 pixel if the
    homography is computed correctly.
    """
    print("\n🔍 Round-trip validation (pixel → world → pixel):")
    
    np.random.seed(42)
    
    for cam_id, cam in sorted(cameras_processed.items()):
        H = cam["H"]
        H_inv = cam["H_inv"]
        
        errors = []
        for _ in range(n_test_points):
            u = np.random.uniform(100, IMG_W - 100)
            v = np.random.uniform(100, IMG_H - 100)
            
            # pixel → world
            X, Y = pixel_to_world(H_inv, u, v)
            
            if abs(X) > 1e5 or abs(Y) > 1e5:
                continue
            
            # world → pixel
            u2, v2 = world_to_pixel(H, X, Y)
            
            err = np.sqrt((u - u2)**2 + (v - v2)**2)
            errors.append(err)
        
        if errors:
            mean_err = np.mean(errors)
            max_err = np.max(errors)
            status = "✅" if max_err < 1.0 else "⚠️"
            print(f"  {status} {cam_id}: mean={mean_err:.6f}px, max={max_err:.6f}px")
        else:
            print(f"  ⚠️ {cam_id}: all test points projected to infinity")


def validate_with_grid(cameras_processed: dict):
    """
    Validation: project WILDTRACK ground grid into each camera view.
    
    This replicates what the official WILDTRACK toolkit does —
    generating ground points and projecting them via cv2.projectPoints().
    We compare our homography-based projection with the direct method.
    """
    print("\n🔍 Grid projection validation (H vs cv2.projectPoints):")
    
    # Sample a sparse grid of ground points
    test_points = []
    for x in np.arange(GROUND_X_MIN, GROUND_X_MAX, 500):  # every 5m
        for y in np.arange(GROUND_Y_MIN, GROUND_Y_MAX, 500):
            test_points.append([x, y, 0.0])
    
    test_points = np.array(test_points, dtype=np.float64)
    
    for cam_id, cam in sorted(cameras_processed.items()):
        H = cam["H"]
        K = cam["K"]
        rvec = cam["rvec"]
        tvec = cam["tvec"]
        dist = cam["dist"]
        
        errors = []
        for pt in test_points:
            # Method 1: Our homography
            u_h, v_h = world_to_pixel(H, pt[0], pt[1])
            
            # Method 2: OpenCV projectPoints (ground truth)
            imgpts, _ = cv2.projectPoints(
                pt.reshape(1, 1, 3), rvec, tvec, K, dist
            )
            u_cv, v_cv = imgpts[0][0]
            
            err = np.sqrt((u_h - u_cv)**2 + (v_h - v_cv)**2)
            errors.append(err)
        
        mean_err = np.mean(errors)
        max_err = np.max(errors)
        status = "✅" if max_err < 2.0 else "⚠️"
        print(f"  {status} {cam_id}: mean={mean_err:.4f}px, max={max_err:.4f}px "
              f"(H vs cv2.projectPoints)")


# ---------------------------------------------------------------------------
# Cache Serialization
# ---------------------------------------------------------------------------
def serialize_cache(cameras_processed: dict, overlaps: dict, 
                    cliques: list, threshold: float) -> dict:
    """
    Build a JSON-serializable cache dictionary.
    
    NumPy arrays are converted to nested lists.
    Shapely polygons are converted to coordinate lists.
    """
    cache = {
        "metadata": {
            "dataset": "WILDTRACK",
            "num_cameras": len(cameras_processed),
            "overlap_threshold": threshold,
            "image_size": [IMG_W, IMG_H],
        },
        "ground_plane": {
            "origin_cm": list(GROUND_ORIGIN_CM),
            "size_cm": list(GROUND_SIZE_CM),
            "bounds": {
                "x_min": GROUND_X_MIN, "x_max": GROUND_X_MAX,
                "y_min": GROUND_Y_MIN, "y_max": GROUND_Y_MAX,
            },
            "grid_step_cm": GRID_STEP_CM,
        },
        "cameras": {},
        "overlap_graph": {
            "threshold": threshold,
            "edges": [],
            "pairwise_overlaps": {},
        },
        "cliques": cliques,
    }
    
    for cam_id, cam in sorted(cameras_processed.items()):
        fov_coords = None
        if cam["fov_polygon"] is not None:
            poly = cam["fov_polygon"]
            if poly.geom_type == 'MultiPolygon':
                # Use the largest sub-polygon
                poly = max(poly.geoms, key=lambda g: g.area)
            fov_coords = [list(coord) for coord in poly.exterior.coords]
        
        cache["cameras"][cam_id] = {
            "K": cam["K"].tolist(),
            "R": cam["R"].tolist(),
            "t": cam["tvec"].flatten().tolist(),
            "rvec": cam["rvec"].flatten().tolist(),
            "P": cam["P"].tolist(),
            "H": cam["H"].tolist(),
            "H_inv": cam["H_inv"].tolist(),
            "fov_polygon": fov_coords,
            "fov_area_m2": cam["fov_polygon"].area / 10000 if cam["fov_polygon"] else 0,
        }
    
    # Overlap graph edges
    for (ci, cj), ratio in sorted(overlaps.items()):
        cache["overlap_graph"]["pairwise_overlaps"][f"{ci}-{cj}"] = round(ratio, 4)
        if ratio >= threshold:
            cache["overlap_graph"]["edges"].append([ci, cj, round(ratio, 4)])
    
    return cache


# ---------------------------------------------------------------------------
# Ablation: Run multiple thresholds
# ---------------------------------------------------------------------------
def run_ablation(overlaps: dict, output_dir: str):
    """
    Run overlap graph construction at multiple thresholds.
    Outputs a summary table for the ablation study.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    thresholds = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = []
    
    print("\n📊 Ablation Study: Overlap Threshold vs Graph Structure")
    print("=" * 75)
    print(f"{'Threshold':>10} | {'Edges':>6} | {'Cliques':>8} | {'Max Clique':>10} | {'Isolated':>8} | Clique List")
    print("-" * 75)
    
    for t in thresholds:
        G, cliques = build_overlap_graph(overlaps, threshold=t)
        n_edges = G.number_of_edges()
        n_cliques = len(cliques)
        max_clique = max(len(c) for c in cliques) if cliques else 0
        isolated = len(list(nx.isolates(G)))
        
        results.append({
            "threshold": t,
            "n_edges": n_edges,
            "n_cliques": n_cliques,
            "max_clique_size": max_clique,
            "isolated_cameras": isolated,
            "cliques": cliques,
        })
        
        clique_str = str([c for c in cliques])
        print(f"  {t:>8.0%} | {n_edges:>6} | {n_cliques:>8} | {max_clique:>10} | "
              f"{isolated:>8} | {clique_str}")
    
    print("=" * 75)
    
    # Plot ablation results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ts = [r["threshold"] for r in results]
    ax1.plot(ts, [r["n_edges"] for r in results], 'o-', color='#FF6B6B', linewidth=2, markersize=8)
    ax1.set_xlabel("Overlap Threshold", fontsize=12)
    ax1.set_ylabel("Number of Graph Edges", fontsize=12)
    ax1.set_title("Edges vs Threshold", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(ts)
    ax1.set_xticklabels([f"{t:.0%}" for t in ts], rotation=45)
    
    ax2.plot(ts, [r["max_clique_size"] for r in results], 's-', color='#4ECDC4', linewidth=2, markersize=8, label='Max Clique Size')
    ax2.plot(ts, [r["n_cliques"] for r in results], 'D-', color='#45B7D1', linewidth=2, markersize=8, label='Num Cliques')
    ax2.plot(ts, [r["isolated_cameras"] for r in results], '^-', color='#FFEAA7', linewidth=2, markersize=8, label='Isolated Cameras')
    ax2.set_xlabel("Overlap Threshold", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Graph Structure vs Threshold", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(ts)
    ax2.set_xticklabels([f"{t:.0%}" for t in ts], rotation=45)
    
    plt.tight_layout()
    ablation_path = os.path.join(output_dir, "ablation_overlap_threshold.png")
    plt.savefig(ablation_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Ablation plot saved to: {ablation_path}")
    
    # Save ablation results as JSON
    ablation_json = os.path.join(output_dir, "ablation_results.json")
    with open(ablation_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Ablation results saved to: {ablation_json}")
    
    return results


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_phase0(calib_dir: str, output_dir: str, threshold: float = 0.05,
               ablation: bool = False):
    """
    Execute the full Phase 0 offline calibration pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  PHASE 0: Offline Calibration Setup")
    print("=" * 60)
    
    # ── Step 1: Load raw calibrations ──────────────────────────
    print("\n📁 Step 1: Loading calibration files...")
    cameras = load_all_calibrations(calib_dir)
    
    # ── Step 2: Compute projection geometry ────────────────────
    print("\n📐 Step 2: Computing projection matrices and homographies...")
    cameras_processed = {}
    
    for cam_id, cam in sorted(cameras.items()):
        K = cam["K"]
        rvec = cam["rvec"]
        tvec = cam["tvec"]
        dist = cam["dist"]
        
        # Rodrigues → rotation matrix
        R = compute_rotation_matrix(rvec)
        
        # Projection matrix: P = K × [R|t]
        P = compute_projection_matrix(K, R, tvec)
        
        # Ground homography: H = K × [r1, r2, t]
        H = compute_ground_homography(K, R, tvec)
        H_inv = np.linalg.inv(H)
        
        cameras_processed[cam_id] = {
            **cam,
            "R": R,
            "P": P,
            "H": H,
            "H_inv": H_inv,
            "fov_polygon": None,  # computed next
        }
        
        # Print summary
        det = np.linalg.det(H)
        print(f"  {cam_id}: det(H)={det:.2f}, "
              f"fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
              f"tvec=[{tvec[0,0]:.0f}, {tvec[1,0]:.0f}, {tvec[2,0]:.0f}]")
    
    # ── Step 3: Compute FOV polygons ───────────────────────────
    print("\n🗺️ Step 3: Computing FOV polygons on ground plane...")
    
    for cam_id, cam in sorted(cameras_processed.items()):
        poly = compute_fov_polygon(cam["H"])
        cameras_processed[cam_id]["fov_polygon"] = poly
        
        if poly is not None:
            area_m2 = poly.area / 10000  # cm² → m²
            bounds = poly.bounds  # (minx, miny, maxx, maxy)
            print(f"  {cam_id}: area={area_m2:.1f} m², "
                  f"bounds=({bounds[0]:.0f}, {bounds[1]:.0f}) → ({bounds[2]:.0f}, {bounds[3]:.0f}) cm")
        else:
            print(f"  {cam_id}: ⚠️ degenerate FOV polygon")
    
    # ── Step 4: Compute pairwise overlaps ──────────────────────
    print("\n🔗 Step 4: Computing pairwise FOV overlaps...")
    
    fov_polygons = {cid: cam["fov_polygon"] for cid, cam in cameras_processed.items()}
    overlaps = compute_pairwise_overlaps(fov_polygons)
    
    print("  Pairwise overlap ratios:")
    for (ci, cj), ratio in sorted(overlaps.items()):
        marker = "🟢" if ratio >= threshold else "⚪"
        print(f"    {marker} {ci} ↔ {cj}: {ratio:.2%}")
    
    # ── Step 5: Build overlap graph & cliques ──────────────────
    print(f"\n🕸️ Step 5: Building overlap graph (threshold={threshold:.0%})...")
    
    G, cliques = build_overlap_graph(overlaps, threshold=threshold)
    
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Maximal cliques ({len(cliques)}):")
    for i, clique in enumerate(cliques):
        print(f"    Clique {i+1}: {clique}")
    
    # ── Step 6: Validation ─────────────────────────────────────
    validate_round_trip(cameras_processed)
    validate_with_grid(cameras_processed)
    
    # ── Step 7: Ablation study (optional) ──────────────────────
    if ablation:
        run_ablation(overlaps, output_dir)
    
    # ── Step 8: Visualizations ─────────────────────────────────
    print("\n🎨 Step 8: Generating visualizations...")
    
    visualize_fov_bev(
        fov_polygons, overlaps, cliques, threshold,
        os.path.join(output_dir, "fov_bev.png")
    )
    
    visualize_overlap_matrix(
        overlaps,
        os.path.join(output_dir, "overlap_matrix.png")
    )
    
    visualize_per_camera_overlaps(
        cameras_processed, fov_polygons,
        output_dir
    )
    
    # ── Step 9: Cache results ──────────────────────────────────
    print("\n💾 Step 9: Saving calibration cache...")
    
    cache = serialize_cache(cameras_processed, overlaps, cliques, threshold)
    cache_path = os.path.join(output_dir, "calibration_cache.json")
    
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    
    print(f"✅ Cache saved to: {cache_path}")
    
    print("\n" + "=" * 60)
    print("  Phase 0 Complete!")
    print("=" * 60)
    
    return cache


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 0: Offline Calibration Setup for WILDTRACK multi-camera tracking"
    )
    parser.add_argument(
        "--calib_dir", type=str,
        default=os.path.join(BASE_DIR, "datasets", "Wildtrack", "calibrations"),
        help="Path to the WILDTRACK calibration directory (default: %(default)s)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(BASE_DIR, "output"),
        help="Directory to save outputs (default: %(default)s)"
    )
    parser.add_argument(
        "--threshold", type=float,
        default=0.05,
        help="FOV overlap threshold for graph edges (default: %(default)s = 5%%)"
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run ablation study across multiple overlap thresholds"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    run_phase0(
        calib_dir=args.calib_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        ablation=args.ablation,
    )
