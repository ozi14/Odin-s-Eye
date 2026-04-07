import cv2
import glob
import os

def create_video(img_pattern, output_path, fps=10):
    images = sorted(glob.glob(img_pattern))
    if not images:
        print(f"No images found for pattern: {img_pattern}")
        return
        
    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    
    # Ensure dimensions are divisible by 2 for mp4v encoder
    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1
    
    # Use mp4v codec for standard mp4 video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Generating video: {output_path} ({len(images)} frames)")
    for img_path in images:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (width, height))
        video.write(img)
        
    cv2.destroyAllWindows()
    video.release()
    print("Done.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    vis_dir = os.path.join(BASE_DIR, "output", "tracking_vis")
    create_video(os.path.join(vis_dir, "grid_*.jpg"), os.path.join(vis_dir, "camera_grid_tracking.mp4"))
    create_video(os.path.join(vis_dir, "bev_*.png"), os.path.join(vis_dir, "bev_tracking.mp4"))
