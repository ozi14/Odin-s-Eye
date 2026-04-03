"""
11_visualize_global.py — Renders Global IDs onto the BEV grid
"""
import os
import sys
import json
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAMERA_IDS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
CAM_COLORS = {
    "C1": (77, 255, 106), "C2": (255, 210, 0), "C3": (238, 104, 123),
    "C4": (71, 179, 255), "C5": (160, 229, 0), "C6": (180, 105, 255), "C7": (235, 206, 135),
}
GROUND_X_MIN, GROUND_X_MAX = -300.0, 3300.0
GROUND_Y_MIN, GROUND_Y_MAX = -90.0, 1110.0

def _get_random_color(seed: str):
    np.random.seed(hash(seed) % (2**32))
    return tuple(map(int, np.random.randint(50, 255, size=3)))

def render_bev():
    global_dir = os.path.join(BASE_DIR, "output", "global_results")
    vis_dir = os.path.join(BASE_DIR, "output", "global_vis")
    os.makedirs(vis_dir, exist_ok=True)
    
    json_files = sorted(glob.glob(os.path.join(global_dir, "frame_*_global.json")))
    print(f"Rending BEV for {len(json_files)} frames...")
    
    for idx, json_path in enumerate(json_files):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        
        ax.add_patch(plt.Rectangle(
            (GROUND_X_MIN, GROUND_Y_MIN), 3600, 1200,
            linewidth=1.5, edgecolor='#4a8a4a', facecolor='#1a2a1a',
            linestyle='--', alpha=0.5
        ))
        
        for g_trk in data.get("global_tracks", []):
            X, Y = g_trk["world_xy"]
            gid = g_trk["global_id"]
            color_rgb = "#{:02x}{:02x}{:02x}".format(*reversed(_get_random_color(gid)))
            
            cams = ",".join(g_trk["cameras_present"])
            
            ax.scatter(X, Y, color=color_rgb, s=120, zorder=5,
                      edgecolors='white', linewidth=1.0)
            ax.text(X + 30, Y + 30, f"{gid} ({cams})",
                   fontsize=8, color=color_rgb, alpha=1.0, zorder=6, fontweight='bold')
                   
        ax.set_xlim(GROUND_X_MIN - 100, GROUND_X_MAX + 100)
        ax.set_ylim(GROUND_Y_MIN - 100, GROUND_Y_MAX + 100)
        ax.set_aspect('equal')
        ax.set_title(f"Frame {data['frame_id']} — Global Identites (Cross-Camera)", 
                    fontsize=14, color='white', fontweight='bold')
                    
        out_path = os.path.join(vis_dir, f"global_bev_{data['frame_id']}.png")
        plt.savefig(out_path, dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)

if __name__ == "__main__":
    render_bev()
